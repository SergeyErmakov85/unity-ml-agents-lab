"""MA-POCA — MultiAgent POsthumous Credit Assignment. Реализация с нуля.

Задача, которую решает метод
----------------------------
Команда забила гол. Награда одна на всех, а игроков двое: один вёл мяч,
второй стоял в углу. Обычный policy gradient повысит вероятность **всех**
действий обоих игроков одинаково — включая бесполезное стояние в углу.
Это и есть проблема распределения заслуги (credit assignment) внутри группы.

Наивные решения не работают:

* **делить награду поровну** — ровно то, что описано выше;
* **давать каждому личную награду** — тогда игроки перестают быть командой:
  каждый начинает тянуть мяч на себя, потому что пас соперника по воротам
  партнёра для него ничем не отличается от промаха;
* **обучать одного «командного» агента, управляющего всеми** — исполнение
  становится централизованным, и в Unity такой агент не разворачивается:
  каждый игрок в сцене видит только своё наблюдение.

Ответ MA-POCA: **CTDE** — централизованное обучение, децентрализованное
исполнение. Актор видит только своё наблюдение (и уходит в ONNX как есть),
критик во время обучения видит наблюдения всей команды.

Контрфактический базлайн
------------------------
Ключевая идея. Обычное преимущество ``A = G − V(s)`` отвечает на вопрос
«насколько ход событий оказался лучше ожидаемого». Для команды этого мало:
вопрос стоит иначе — «насколько лучше стало **из-за игрока i**».

MA-POCA отвечает на него контрфактическим базлайном::

    Q_i = Q(o_1..o_n, a_1..a_n **без** a_i)

то есть ожидаемым возвратом команды, если известны действия всех, кроме
i-го. Разность::

    A_i = G^λ − Q_i

и есть вклад именно i-го игрока: всё, что объясняется действиями остальных,
уже сидит в ``Q_i`` и вычитается. Игрок в углу получает ``A_i ≈ 0``
и не усиливается; тот, кто забил, — большое ``A_i``.

Почему внимание, а не склейка
-----------------------------
И ``V``, и ``Q_i`` принимают **множество** сущностей переменного размера
и обязаны не зависеть от порядка игроков. Это ровно то, что даёт
self-attention (:mod:`labrl.nets.attention`). Отсюда же «posthumous»
в названии метода: игрок, закончивший эпизод раньше остальных, просто
исчезает из множества — маской, — и его вклад в уже случившиеся шаги
при этом не теряется.

Что уходит в ONNX
-----------------
Только актор. ``V`` и ``Q_i`` — строительные леса обучения; в Unity каждый
игрок исполняет свою политику по своему наблюдению, и централизованных
входов там взять неоткуда. Это и есть практический смысл буквы «D»
в CTDE.

Как обновляется политика
------------------------
Обрезкой отношения правдоподобий, как в PPO (:mod:`labrl.algos.ppo`):
метод от этого не перестаёт быть MA-POCA — новизна в **критике
и базлайне**, а не в способе шага по политике. Штатный тренер ML-Agents
поступает так же.

Источник: Cohen et al., «On the Use and Misuse of Absorbing States
in Multi-agent Reinforcement Learning» (2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labrl.buffers.group_rollout import GroupRolloutBatch
from labrl.nets.attention import EntityEncoder, ResidualSelfAttention, masked_mean


@dataclass
class MAPOCAConfig:
    """Все гиперпараметры метода. «Магических чисел» внутри методов нет (8.6)."""

    #: Коэффициент дисконтирования, безразмерный, [0, 1).
    gamma: float = 0.99
    #: Параметр λ для λ-возврата, [0, 1].
    gae_lambda: float = 0.95
    #: Скорость обучения общего оптимизатора Adam.
    learning_rate: float = 3e-4
    #: Ширина обрезки ε отношения правдоподобий (как в PPO).
    clip_range: float = 0.2
    #: Сколько раз пройти по собранному роллауту.
    epochs: int = 3
    #: Размер мини-батча в шагах команды (не в переходах отдельных агентов).
    minibatch_size: int = 128
    #: Вес ошибки централизованного критика V.
    value_coef: float = 0.5
    #: Вес ошибки контрфактического базлайна Q_i.
    baseline_coef: float = 0.5
    #: Вес бонуса за энтропию. У дискретной политики с тремя ветками
    #: энтропия на старте равна 3·ln3 ≈ 3.3 нат, поэтому коэффициент
    #: заметно меньше, чем у одноветочных сред.
    entropy_coef: float = 0.005
    #: Максимальная норма градиента; 0 — не клипировать.
    max_grad_norm: float = 0.5
    #: Нормировать ли преимущества по мини-батчу.
    normalize_advantage: bool = True
    #: Порог приближённой KL для досрочного выхода из эпох; 0 — не выходить.
    target_kl: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda должна быть в [0, 1], получено {self.gae_lambda}")
        if self.clip_range <= 0.0:
            raise ValueError(f"clip_range должен быть > 0, получено {self.clip_range}")
        if self.epochs <= 0:
            raise ValueError(f"epochs должно быть > 0, получено {self.epochs}")
        if self.minibatch_size <= 0:
            raise ValueError(f"minibatch_size должен быть > 0, получено {self.minibatch_size}")


class CentralizedCritic(nn.Module):
    """``V(o_1 … o_n)`` — ожидаемый возврат **команды**.

    Множество наблюдений кодируется в общее пространство, проходит через
    self-attention и симметрично сворачивается средним. Результат не зависит
    ни от порядка игроков, ни от их числа.

    Args:
        obs_dim: размерность наблюдения одного игрока.
        embed_dim: размерность представления сущности.
        num_heads: число голов внимания.
        hidden_dim: ширина скрытых слоёв кодировщика и головы.
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_encoder = EntityEncoder(obs_dim, embed_dim, hidden_dim)
        self.attention = ResidualSelfAttention(embed_dim, num_heads)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Args: ``obs`` ``(B, n, obs_dim)``, ``active`` ``(B, n)``. Returns: ``(B,)``."""
        entities = self.obs_encoder(obs)
        attended = self.attention(entities, active)
        return self.head(masked_mean(attended, active)).squeeze(-1)


class CounterfactualBaseline(nn.Module):
    """``Q_i(o_1 … o_n, a_{−i})`` — возврат команды при известных действиях
    всех, **кроме** i-го игрока.

    Множество сущностей неоднородно, и в этом вся суть:

    * i-й игрок входит **только наблюдением** — его действие скрыто;
    * остальные входят парой «наблюдение + действие».

    Поэтому кодировщиков два, с разными входными размерностями, но с общим
    выходным пространством: внимание обязано сравнивать сущности между собой.

    Действия подаются в one-hot по каждой ветке. Индекс ветки сам по себе —
    не число: «повернуть налево» не находится «между» «стоять» и «повернуть
    направо», и подавать 0/1/2 как скаляр означало бы навязать сети
    несуществующий порядок.

    Args:
        obs_dim: размерность наблюдения одного игрока.
        branches: размеры дискретных веток действия.
        embed_dim, num_heads, hidden_dim: как у :class:`CentralizedCritic`.
    """

    def __init__(
        self,
        obs_dim: int,
        branches: Sequence[int],
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.branches = tuple(int(b) for b in branches)
        action_dim = sum(self.branches)

        self.obs_encoder = EntityEncoder(obs_dim, embed_dim, hidden_dim)
        self.obs_action_encoder = EntityEncoder(obs_dim + action_dim, embed_dim, hidden_dim)
        self.attention = ResidualSelfAttention(embed_dim, num_heads)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def one_hot_actions(self, action: torch.Tensor) -> torch.Tensor:
        """``(B, n, num_branches)`` индексов -> ``(B, n, sum(branches))`` one-hot."""
        parts = [
            F.one_hot(action[..., branch].long(), num_classes=size).float()
            for branch, size in enumerate(self.branches)
        ]
        return torch.cat(parts, dim=-1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """``Q_i`` для **каждого** игрока сразу.

        Args:
            obs: ``(B, n, obs_dim)``.
            action: ``(B, n, num_branches)`` — индексы веток.
            active: ``(B, n)``.

        Returns:
            ``(B, n)`` — по одному значению на игрока.

        Реализация считает все ``n`` базлайнов одним прогоном: батч
        разворачивается в ``(B·n, n, …)``, где строка ``i`` содержит
        сущности с точки зрения i-го игрока. Это в ``n`` раз дороже одного
        прогона, но не в ``n`` раз медленнее цикла на Python по игрокам —
        и, что важнее, все ``n`` базлайнов оказываются в одном графе,
        поэтому обратный проход тоже один.
        """
        batch, count, obs_dim = obs.shape

        actions_1h = self.one_hot_actions(action)                       # (B, n, A)
        obs_only = self.obs_encoder(obs)                                # (B, n, E)
        obs_with_action = self.obs_action_encoder(
            torch.cat([obs, actions_1h], dim=-1)
        )                                                               # (B, n, E)

        # Разворачиваем: строка i — множество сущностей глазами i-го игрока.
        obs_only_x = obs_only.unsqueeze(1).expand(batch, count, count, -1)
        obs_act_x = obs_with_action.unsqueeze(1).expand(batch, count, count, -1)

        # eye[i, j] = 1, если j == i: там берётся сущность БЕЗ действия.
        eye = torch.eye(count, device=obs.device, dtype=obs.dtype).view(1, count, count, 1)
        entities = eye * obs_only_x + (1.0 - eye) * obs_act_x           # (B, n, n, E)

        embed = entities.shape[-1]
        entities = entities.reshape(batch * count, count, embed)
        mask = active.unsqueeze(1).expand(batch, count, count).reshape(batch * count, count)

        attended = self.attention(entities, mask)
        pooled = masked_mean(attended, mask)
        return self.head(pooled).view(batch, count)


class MAPOCA:
    """MA-POCA: децентрализованный актор, централизованный критик и
    контрфактический базлайн.

    Args:
        policy_net: актор — дискретная политика с несколькими ветками
            (:class:`labrl.nets.categorical_policy.MultiBranchCategoricalPolicy`).
            Архитектуру задаёт пользователь (требование 8.4).
        value_net: :class:`CentralizedCritic`.
        baseline_net: :class:`CounterfactualBaseline`.
        cfg: гиперпараметры.
        device: устройство вычислений.
        seed: сид перемешивания мини-батчей.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        baseline_net: nn.Module,
        cfg: MAPOCAConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or MAPOCAConfig()
        self.device = torch.device(device)

        self.policy_net = policy_net.to(self.device)
        self.value_net = value_net.to(self.device)
        self.baseline_net = baseline_net.to(self.device)
        self.optimizer = torch.optim.Adam(self._parameters(), lr=self.cfg.learning_rate)

        self._rng = np.random.default_rng(seed)
        self.updates = 0

    def _parameters(self) -> list[nn.Parameter]:
        return (
            list(self.policy_net.parameters())
            + list(self.value_net.parameters())
            + list(self.baseline_net.parameters())
        )

    # --- взаимодействие со средой ---------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Сэмплирует действия для батча наблюдений **отдельных** игроков.

        Args:
            obs: ``(B, obs_dim)``.

        Returns:
            ``(действия (B, num_branches) из индексов, log π (B,))``.

        Актор децентрализован: он ничего не знает о команде, и здесь это
        видно буквально — на вход идёт только наблюдение игрока.
        """
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        action, log_prob = self.policy_net.sample(tensor)
        return action.cpu().numpy().astype(np.int64), log_prob.cpu().numpy()

    @torch.no_grad()
    def deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Наиболее вероятное действие — то же, что даёт
        ``deterministic_discrete_actions`` в графе ONNX."""
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.policy_net.greedy(tensor).cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def team_value(self, obs: np.ndarray, active: np.ndarray) -> np.ndarray:
        """``V`` команды для батча состояний. ``(B, n, obs_dim)``, ``(B, n)`` -> ``(B,)``."""
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        active_t = torch.as_tensor(np.asarray(active, dtype=np.float32), device=self.device)
        return self.value_net(obs_t, active_t).cpu().numpy()

    # --- обучение --------------------------------------------------------

    def update(self, batch: GroupRolloutBatch) -> dict[str, float]:
        """Единственное место, где меняются параметры (требование 8.7)."""
        size = len(batch)
        obs = torch.as_tensor(batch.obs, device=self.device)
        action = torch.as_tensor(batch.action, device=self.device)
        active = torch.as_tensor(batch.active, device=self.device)
        old_log_prob = torch.as_tensor(batch.log_prob, device=self.device)
        returns = torch.as_tensor(batch.returns, device=self.device)

        stats: dict[str, float] = {}
        clip_fractions: list[float] = []
        epochs_done = 0
        stop = False

        for _ in range(self.cfg.epochs):
            order = self._rng.permutation(size)
            for start in range(0, size, self.cfg.minibatch_size):
                index = torch.as_tensor(
                    order[start : start + self.cfg.minibatch_size], device=self.device
                )
                if index.numel() < 2:
                    continue

                mb_obs = obs[index]                # (b, n, obs_dim)
                mb_action = action[index]          # (b, n, branches)
                mb_active = active[index]          # (b, n)
                mb_returns = returns[index]        # (b,)
                mb_old_log_prob = old_log_prob[index]

                b, n, obs_dim = mb_obs.shape

                value = self.value_net(mb_obs, mb_active)                     # (b,)
                baseline = self.baseline_net(mb_obs, mb_action, mb_active)    # (b, n)

                # Контрфактическое преимущество каждого игрока. detach:
                # базлайн обучается собственной функцией потерь, а через
                # преимущество градиент к нему идти не должен — иначе метод
                # начнёт занижать базлайн, чтобы «увеличить» преимущество.
                advantage = mb_returns.unsqueeze(-1) - baseline.detach()      # (b, n)

                if self.cfg.normalize_advantage:
                    # Нормировка только по существующим игрокам: включив
                    # отсутствующих, мы усреднили бы нули и сдвинули масштаб.
                    weight = mb_active
                    count = weight.sum().clamp(min=1.0)
                    mean = (advantage * weight).sum() / count
                    var = (((advantage - mean) ** 2) * weight).sum() / count
                    advantage = (advantage - mean) / (var.sqrt() + 1e-8)

                # Актор смотрит на игроков как на независимые примеры:
                # веса общие, наблюдения свои.
                flat_obs = mb_obs.reshape(b * n, obs_dim)
                flat_action = mb_action.reshape(b * n, mb_action.shape[-1])
                log_prob = self.policy_net.log_prob(flat_obs, flat_action).view(b, n)

                ratio = torch.exp(log_prob - mb_old_log_prob)
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range
                ) * advantage
                policy_loss = -_masked_mean(torch.min(unclipped, clipped), mb_active)

                value_loss = F.mse_loss(value, mb_returns)
                # Базлайн учится той же цели, что и V: разница между ними —
                # только в том, что базлайн видит действия остальных игроков.
                baseline_loss = _masked_mean(
                    (baseline - mb_returns.unsqueeze(-1)) ** 2, mb_active
                )
                # Энтропия — только по существующим игрокам. Строка
                # отсутствующего игрока заполнена нулями, и её энтропия
                # (политика на нулевом наблюдении) не имеет отношения ни
                # к какому реальному состоянию: включив её в бонус, мы
                # заставили бы политику поддерживать разведку в точке,
                # которую среда никогда не покажет.
                present = mb_active.reshape(b * n) > 0
                entropy = self.policy_net.entropy(flat_obs[present])

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.baseline_coef * baseline_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._parameters(),
                    self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf"),
                )
                self.optimizer.step()
                self.updates += 1

                with torch.no_grad():
                    approx_kl = _masked_mean(
                        (ratio - 1.0) - (log_prob - mb_old_log_prob), mb_active
                    )
                    clip_fractions.append(
                        float(
                            _masked_mean(
                                ((ratio - 1.0).abs() > self.cfg.clip_range).float(), mb_active
                            ).item()
                        )
                    )

                stats = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "baseline_loss": float(baseline_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                    "grad_norm": float(grad_norm),
                    # Разброс базлайна по игрокам — прямая мера того, различает
                    # ли метод их вклад вообще. Ноль означает, что критик
                    # игнорирует действия, и MA-POCA выродился в обычный PPO.
                    "baseline_spread": float(
                        (baseline.max(dim=1).values - baseline.min(dim=1).values).mean().item()
                    ),
                }

                if self.cfg.target_kl > 0.0 and stats["approx_kl"] > self.cfg.target_kl:
                    stop = True
                    break

            epochs_done += 1
            if stop:
                break

        with torch.no_grad():
            predicted = self.value_net(obs, active).cpu().numpy()

        stats.update(
            {
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "explained_variance": _explained_variance(predicted, batch.returns),
                "epochs_done": float(epochs_done),
                "updates": float(self.updates),
            }
        )
        return stats

    def set_learning_rate(self, learning_rate: float) -> None:
        """Меняет шаг Adam — для расписания скорости обучения (T-13)."""
        if learning_rate < 0.0:
            raise ValueError(f"learning_rate должен быть >= 0, получено {learning_rate}")
        self.cfg.learning_rate = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    # --- сохранение и экспорт -------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "baseline_net": self.baseline_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(state["policy_net"])
        self.value_net.load_state_dict(state["value_net"])
        self.baseline_net.load_state_dict(state["baseline_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))

    def policy_module(self) -> nn.Module:
        """Модуль, экспортируемый в ONNX: только актор.

        Централизованный критик и базлайн в Unity не нужны и не могут быть
        нужны: у игрока в сцене нет наблюдений партнёра. Это буква «D»
        в CTDE, выраженная одной строкой.
        """
        return self.policy_net.eval()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Среднее по существующим элементам. Скаляр."""
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def _explained_variance(predicted: np.ndarray, target: np.ndarray) -> float:
    """Доля дисперсии цели, объяснённая критиком. 1 — идеально, 0 — бесполезен."""
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    variance = target.var()
    if variance < 1e-12:
        return 0.0
    return float(1.0 - (target - predicted).var() / variance)
