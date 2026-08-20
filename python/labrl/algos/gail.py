"""GAIL — Generative Adversarial Imitation Learning. Реализация с нуля.

Что чинит GAIL по сравнению с BC
--------------------------------
BC (:mod:`labrl.algos.bc`) учится только на состояниях, которые посещал
эксперт. Первая же ошибка выводит политику за пределы этих состояний, и там
она ничем не обусловлена — ошибки накапливаются (сдвиг распределения,
Ross & Bagnell, 2010).

GAIL смотрит на задачу иначе: **научимся отличать поведение политики
от поведения эксперта, и пусть неотличимость будет наградой**. Тогда сигнал
есть в любом состоянии, включая те, куда эксперт никогда не заходил, —
там дискриминатор уверенно говорит «это не эксперт», и политика получает
низкую награду.

Устройство
----------
Две обучаемые части и обычный RL между ними:

1. **Дискриминатор** ``D_ψ(s, a) ∈ (0, 1)`` — вероятность того, что пара
   пришла от эксперта. Обучается бинарной кросс-энтропией::

       L_D = − E_эксперт[ log D(s,a) ] − E_политика[ log(1 − D(s,a)) ]

2. **Награда для политики**::

       r_GAIL(s, a) = − log(1 − D(s,a))

   Она положительна и растёт, когда дискриминатор считает пару экспертной.
   Форма выбрана не произвольно: `−log(1−D)` не ограничена сверху и даёт
   сильный градиент, когда политика уже похожа на эксперта, тогда как
   `log D` насыщается ровно в этот момент. Обратная сторона — награда всегда
   положительна, и агент имеет мотив **тянуть эпизод**, а не заканчивать его;
   в средах «дойди до цели» это лечится подмешиванием награды среды
   (``env_reward_weight``).

3. **Политика** обучается любым RL-методом на этой награде. Здесь —
   :class:`labrl.algos.ppo_discrete.PPODiscrete`: GAIL задаёт награду,
   а не способ оптимизации.

Что делает обучение устойчивым
------------------------------
Дискриминатор — противник политики, и, как всякое состязательное обучение,
эта пара легко расходится. Три средства, все три реализованы:

* **сглаживание меток** (``label_smoothing``): цель эксперта не 1.0, а 0.9.
  Идеально уверенный дискриминатор даёт нулевой градиент политике;
* **штраф на градиент** (``gradient_penalty``, Gulrajani et al., 2017):
  ограничивает липшицеву константу дискриминатора и не даёт ему «обрубать»
  награду;
* **редкие обновления** (``updates_per_batch``): дискриминатор, обучающийся
  так же часто, как политика, выигрывает состязание слишком быстро.

Источник: Ho & Ermon, «Generative Adversarial Imitation Learning» (2016).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labrl.envs.demos import Demonstrations
from labrl.nets.mlp import ACTIVATIONS, build_mlp


@dataclass
class GAILConfig:
    """Все гиперпараметры дискриминатора. «Магических чисел» внутри нет (8.6)."""

    #: Скорость обучения Adam дискриминатора. Обычно ниже, чем у политики:
    #: слишком быстрый дискриминатор выигрывает состязание и обнуляет сигнал.
    learning_rate: float = 1e-4
    #: Размер мини-батча (столько же берётся от эксперта и от политики).
    batch_size: int = 256
    #: Сколько шагов дискриминатора на один роллаут политики.
    updates_per_batch: int = 2
    #: Метка эксперта вместо 1.0. Идеально уверенный дискриминатор даёт
    #: нулевой градиент политике.
    label_smoothing: float = 0.1
    #: Вес штрафа на норму градиента дискриминатора. 0 — выключено.
    gradient_penalty: float = 10.0
    #: Максимальная норма градиента; 0 — не клипировать.
    max_grad_norm: float = 1.0
    #: Доля награды среды в итоговой награде политики.
    #: 0 — чистый GAIL (награда среды не используется вовсе);
    #: 1 — чистый RL. Промежуточные значения лечат склонность GAIL
    #: тянуть эпизод (награда `−log(1−D)` всегда положительна).
    env_reward_weight: float = 0.0
    #: Масштаб награды дискриминатора.
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(
                f"label_smoothing должен быть в [0, 0.5), получено {self.label_smoothing}"
            )
        if not 0.0 <= self.env_reward_weight <= 1.0:
            raise ValueError(
                f"env_reward_weight должен быть в [0, 1], получено {self.env_reward_weight}"
            )
        if self.updates_per_batch <= 0:
            raise ValueError(
                f"updates_per_batch должен быть > 0, получено {self.updates_per_batch}"
            )


class Discriminator(nn.Module):
    """``D(s, a) -> логит`` того, что пара пришла от эксперта.

    Действие подаётся в one-hot по каждой ветке: индекс ветки — не число,
    и «повернуть налево» не находится «между» «стоять» и «повернуть направо».

    Args:
        obs_dim: размерность наблюдения.
        branches: размеры дискретных веток.
        hidden_sizes: скрытые слои.
        activation: имя активации из :data:`labrl.nets.mlp.ACTIVATIONS`.

    Возвращается **логит**, а не вероятность: и функция потерь
    (`binary_cross_entropy_with_logits`), и награда (`softplus`) численно
    устойчивее, когда сигмоида не вычисляется отдельно.
    """

    def __init__(
        self,
        obs_dim: int,
        branches: Sequence[int],
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation должна быть одной из {sorted(ACTIVATIONS)}")
        self.branches = tuple(int(b) for b in branches)
        self.obs_dim = int(obs_dim)
        self.net = build_mlp(self.obs_dim + sum(self.branches), 1, hidden_sizes, activation)

    def one_hot(self, action: torch.Tensor) -> torch.Tensor:
        """``(B, num_branches)`` индексов -> ``(B, sum(branches))`` one-hot."""
        parts = [
            F.one_hot(action[:, branch].long(), num_classes=size).float()
            for branch, size in enumerate(self.branches)
        ]
        return torch.cat(parts, dim=-1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Логит. ``(B, obs_dim)``, ``(B, num_branches)`` -> ``(B,)``."""
        return self.net(torch.cat([obs, self.one_hot(action)], dim=-1)).squeeze(-1)


class GAIL:
    """Дискриминатор GAIL и награда, которую он порождает.

    Класс **не обучает политику**: он выдаёт награду, а политику двигает
    любой RL-метод (в лаборатории — `PPODiscrete`). Разделение сознательное:
    GAIL — это способ получить награду, а не способ оптимизации, и смешивать
    их в одном классе значило бы прятать это различие.

    Args:
        discriminator: :class:`Discriminator` либо любой модуль с той же
            сигнатурой ``forward(obs, action) -> (B,)``.
        demos: демонстрации эксперта.
        cfg: гиперпараметры.
        device: устройство вычислений.
        seed: сид выборки мини-батчей.
    """

    def __init__(
        self,
        discriminator: nn.Module,
        demos: Demonstrations,
        cfg: GAILConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or GAILConfig()
        self.device = torch.device(device)
        self.discriminator = discriminator.to(self.device)
        self.demos = demos
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.cfg.learning_rate
        )
        self._rng = np.random.default_rng(seed)
        self.updates = 0

    # --- награда ---------------------------------------------------------

    @torch.no_grad()
    def reward(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """``r_GAIL = − log(1 − D(s,a))`` для батча пар. ``(B,)``.

        Через softplus: ``− log(1 − σ(x)) = softplus(x)``. Тождество точное,
        и оно избавляет от вычитания близких к единице чисел, на котором
        прямая формула теряет точность именно тогда, когда политика уже
        похожа на эксперта.
        """
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        action_t = torch.as_tensor(np.asarray(action, dtype=np.int64), device=self.device)
        logits = self.discriminator(obs_t, action_t)
        return (F.softplus(logits) * self.cfg.reward_scale).cpu().numpy()

    def mixed_reward(self, gail_reward: np.ndarray, env_reward: np.ndarray) -> np.ndarray:
        """Смесь награды дискриминатора и награды среды.

        При ``env_reward_weight = 0`` среда не участвует вовсе — это чистый
        GAIL, и он честно показывает, на что способна имитация без функции
        награды. Положительный вес лечит склонность GAIL тянуть эпизод:
        награда `−log(1−D)` всегда положительна, и «жить дольше» само по себе
        выгодно.
        """
        w = self.cfg.env_reward_weight
        return (1.0 - w) * np.asarray(gail_reward) + w * np.asarray(env_reward)

    # --- обучение --------------------------------------------------------

    def update(self, batch: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        """Единственное место, где меняются параметры дискриминатора (8.7).

        Args:
            batch: ``(obs, action)`` — пары, собранные **политикой**.
        """
        policy_obs_np, policy_action_np = batch
        stats: dict[str, float] = {}

        for _ in range(self.cfg.updates_per_batch):
            size = min(self.cfg.batch_size, len(policy_obs_np))
            index = self._rng.integers(0, len(policy_obs_np), size=size)
            policy_obs = torch.as_tensor(
                np.asarray(policy_obs_np, dtype=np.float32)[index], device=self.device)
            policy_action = torch.as_tensor(
                np.asarray(policy_action_np, dtype=np.int64)[index], device=self.device)

            expert_obs_np, expert_action_np = self.demos.sample(size, self._rng)
            expert_obs = torch.as_tensor(expert_obs_np, device=self.device)
            expert_action = torch.as_tensor(expert_action_np, device=self.device)

            expert_logits = self.discriminator(expert_obs, expert_action)
            policy_logits = self.discriminator(policy_obs, policy_action)

            # Сглаживание меток: цель эксперта 0.9, а не 1.0. Идеально
            # уверенный дискриминатор даёт нулевой градиент политике.
            expert_target = torch.full_like(expert_logits, 1.0 - self.cfg.label_smoothing)
            policy_target = torch.zeros_like(policy_logits)

            loss = (F.binary_cross_entropy_with_logits(expert_logits, expert_target)
                    + F.binary_cross_entropy_with_logits(policy_logits, policy_target))

            penalty = torch.zeros((), device=self.device)
            if self.cfg.gradient_penalty > 0.0:
                penalty = self._gradient_penalty(expert_obs, expert_action, policy_obs, policy_action)
                loss = loss + self.cfg.gradient_penalty * penalty

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf"),
            )
            self.optimizer.step()
            self.updates += 1

            with torch.no_grad():
                expert_accuracy = (expert_logits > 0).float().mean()
                policy_accuracy = (policy_logits < 0).float().mean()

            stats = {
                "discriminator_loss": float(loss.item()),
                "gradient_penalty": float(penalty.item()),
                # Точность дискриминатора — главная диагностика GAIL.
                # 0.5 означает «политика неотличима от эксперта» (цель),
                # 1.0 — «дискриминатор выиграл, сигнал политике пропал».
                "discriminator_accuracy": float(0.5 * (expert_accuracy + policy_accuracy).item()),
                "expert_accuracy": float(expert_accuracy.item()),
                "policy_accuracy": float(policy_accuracy.item()),
                "discriminator_grad_norm": float(grad_norm),
                "discriminator_updates": float(self.updates),
            }
        return stats

    def _gradient_penalty(
        self,
        expert_obs: torch.Tensor,
        expert_action: torch.Tensor,
        policy_obs: torch.Tensor,
        policy_action: torch.Tensor,
    ) -> torch.Tensor:
        """Штраф ``(‖∇D‖ − 1)²`` на смеси экспертных и политиковых наблюдений.

        Интерполируется **только наблюдение**: действие дискретно, и точка
        «между» двумя действиями смысла не имеет. Это отличие от классического
        WGAN-GP, где интерполируются оба аргумента.
        """
        size = min(expert_obs.shape[0], policy_obs.shape[0])
        alpha = torch.rand(size, 1, device=self.device)
        mixed = (alpha * expert_obs[:size] + (1.0 - alpha) * policy_obs[:size]).requires_grad_(True)

        logits = self.discriminator(mixed, expert_action[:size])
        gradients = torch.autograd.grad(
            outputs=logits.sum(), inputs=mixed, create_graph=True, retain_graph=True,
        )[0]
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    def set_learning_rate(self, learning_rate: float) -> None:
        if learning_rate < 0.0:
            raise ValueError(f"learning_rate должен быть >= 0, получено {learning_rate}")
        self.cfg.learning_rate = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    # --- сохранение ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))
