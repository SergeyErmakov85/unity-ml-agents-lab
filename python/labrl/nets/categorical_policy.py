"""Дискретная политика с несколькими независимыми ветками действий.

Зачем отдельный модуль. До `E08_SoccerArena` дискретные действия встречались
в двух видах: одна ветка у DQN (`labrl.nets.mlp`, Q-значения) и дискретная
часть гибрида (`labrl.nets.hybrid_policy`). Здесь нужен третий, самый простой
случай — **чисто дискретная политика с несколькими ветками**.

Почему ветки, а не одно распределение на все комбинации. Действие футболиста
складывается из трёх независимых решений: ход, движение боком, поворот.
Одно распределение на 3 × 3 × 3 = 27 комбинаций потребовало бы 27 логитов
и учило бы каждую комбинацию отдельно: «вперёд + вправо» не помогало бы
выучить «вперёд + влево». Три независимые ветки дают 9 логитов и общее
представление хода, поворота и смещения::

    π(a | s) = π₀(a₀ | s) · π₁(a₁ | s) · π₂(a₂ | s)

Ровно такую факторизацию использует и ML-Agents: `ActionSpec.MakeDiscrete`
принимает список размеров веток, а контракт ONNX отдаёт `discrete_actions`
формы ``(batch, num_branches)`` — по индексу на ветку
(`docs/04_ONNX_CONTRACT.md`, §2).

**Сеть возвращает логиты, а не действия.** Сэмплирование живёт в алгоритме
(`labrl.algos`) и в обёртке экспорта (`labrl.export.onnx_export`
со ``strategy="categorical"``). Иначе одну и ту же логику пришлось бы
поддерживать в двух местах, и рано или поздно они разошлись бы.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from labrl.nets.mlp import ACTIVATIONS, build_mlp


class MultiBranchCategoricalPolicy(nn.Module):
    """``(B, obs_dim) -> (B, sum(branches))`` — логиты всех веток подряд.

    Args:
        obs_dim: размерность наблюдения (все сенсоры уже склеены).
        branches: размеры дискретных веток, например ``(3, 3, 3)``.
        hidden_sizes: скрытые слои.
        activation: имя активации из :data:`labrl.nets.mlp.ACTIVATIONS`.

    Порядок логитов в выходе — ветка за веткой, без разделителей: первые
    ``branches[0]`` чисел относятся к ветке 0, следующие ``branches[1]`` —
    к ветке 1 и так далее. Именно такой плоский тензор ждёт
    :class:`labrl.export.onnx_export.MLAgentsPolicyWrapper`.
    """

    def __init__(
        self,
        obs_dim: int,
        branches: Sequence[int],
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation должна быть одной из {sorted(ACTIVATIONS)}")
        self.branches = tuple(int(b) for b in branches)
        if not self.branches or any(b < 2 for b in self.branches):
            raise ValueError(f"каждая ветка должна иметь >= 2 действий, получено {self.branches}")

        self.obs_dim = int(obs_dim)
        self.net = build_mlp(self.obs_dim, sum(self.branches), hidden_sizes, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    # --- распределения ---------------------------------------------------

    def split_logits(self, logits: torch.Tensor) -> list[torch.Tensor]:
        """Логиты по веткам: список из ``len(branches)`` тензоров ``(B, branch)``."""
        return list(torch.split(logits, list(self.branches), dim=-1))

    def distributions(self, obs: torch.Tensor) -> list[torch.distributions.Categorical]:
        """По категориальному распределению на ветку."""
        return [
            torch.distributions.Categorical(logits=branch_logits)
            for branch_logits in self.split_logits(self(obs))
        ]

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """``log π(a|s)`` для действия ``(B, num_branches)`` из индексов. ``(B,)``.

        Ветки независимы, поэтому логарифмы складываются.
        """
        total = torch.zeros(obs.shape[0], device=obs.device)
        for branch, distribution in enumerate(self.distributions(obs)):
            total = total + distribution.log_prob(action[:, branch])
        return total

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Средняя по батчу энтропия политики, нат: сумма энтропий веток. Скаляр."""
        total = torch.zeros((), device=obs.device)
        for distribution in self.distributions(obs):
            total = total + distribution.entropy().mean()
        return total

    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Сэмплированное действие ``(B, num_branches)`` и его ``log π``. ``(B,)``.

        Сэмплирование делается PyTorch-генератором. Воспроизводимость
        обеспечивается глобальным сидом (`labrl.utils.seeding.set_global_seed`),
        а не отдельным ``np.random.Generator``: категориальное сэмплирование
        нужно и здесь, и внутри графа ONNX (`torch.multinomial`), и заводить
        для них разные источники случайности значило бы гарантированно
        получить разное поведение Python и Unity.
        """
        actions: list[torch.Tensor] = []
        log_probs = torch.zeros(obs.shape[0], device=obs.device)
        for distribution in self.distributions(obs):
            action = distribution.sample()
            log_probs = log_probs + distribution.log_prob(action)
            actions.append(action.unsqueeze(-1))
        return torch.cat(actions, dim=-1), log_probs

    @torch.no_grad()
    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        """Наиболее вероятное действие каждой ветки — то же, что даёт
        ``deterministic_discrete_actions`` в графе ONNX."""
        return torch.cat(
            [branch.argmax(dim=-1, keepdim=True) for branch in self.split_logits(self(obs))],
            dim=-1,
        )
