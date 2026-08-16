"""REINFORCE — Monte-Carlo policy gradient (Williams, 1992). Реализация с нуля.

Урок 2.1. Это самый прямой способ обучать политику: взять полный эпизод,
посчитать для каждого шага фактический возврат ``G_t`` и сдвинуть параметры
так, чтобы действия с большим возвратом стали вероятнее::

    ∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ],
    G_t = Σ_{k≥t} γ^{k−t} r_k

Никакого критика, никакого бутстрэппинга, никаких целевых сетей — только
логарифм правдоподобия и фактическая награда.

Цена простоты — **дисперсия**. Возврат ``G_t`` включает всё, что случилось
после шага t, в том числе никак с ним не связанное. Оценка градиента получается
несмещённой, но настолько шумной, что без лечения метод почти не сходится.

Лечение — **baseline**::

    ∇_θ J = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t − b(s_t)) ]

Вычитание любой функции состояния не меняет математическое ожидание (это
доказывается в уроке 2.1), но резко уменьшает дисперсию. Лучший выбор —
``b(s) = V^π(s)``, и тогда в скобках оказывается оценка преимущества.
Именно отсюда один шаг до Actor-Critic (`labrl.algos.a2c`): заменить
Monte-Carlo возврат на бутстрэп-оценку — и получится A2C.

Чем этот файл отличается от `a2c.py`
------------------------------------
* ``G_t`` считается по **фактическому** эпизоду до конца, а не бутстрэпом;
* обновление делается по **завершённым эпизодам**, а не по нарезке шагов;
* критик (если включён) — именно baseline: он не участвует в цели,
  а только вычитается.

Гибридное пространство действий
-------------------------------
`E05_FoodCollector` управляется одновременно непрерывным движением и дискретным
переключателем. Части действия независимы, поэтому логарифмы складываются:
``log π(a) = log π_непр(a_c) + Σ_ветки log π_дискр(a_d)``. Ни одна строчка
самого REINFORCE от этого не меняется — меняется только политика.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labrl.export.onnx_export import CONTINUOUS_CLIP
from labrl.nets.gaussian_policy import to_env_action


@dataclass
class ReinforceConfig:
    """Все гиперпараметры метода. «Магических чисел» внутри методов нет (8.6)."""

    #: Коэффициент дисконтирования, безразмерный, [0, 1).
    gamma: float = 0.99
    #: Скорость обучения Adam (общая для политики и baseline).
    learning_rate: float = 3e-4
    #: Вес ошибки baseline в общей функции потерь. 0 — baseline не обучается.
    value_coef: float = 0.5
    #: Вес бонуса за энтропию.
    entropy_coef: float = 0.01
    #: Максимальная норма градиента; 0 — не клипировать.
    max_grad_norm: float = 0.5
    #: Нормировать ли преимущества по батчу. Для REINFORCE это почти
    #: обязательно: масштаб возвратов зависит от среды и меняется по ходу
    #: обучения, а шаг Adam к масштабу чувствителен.
    normalize_advantage: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate должен быть > 0, получено {self.learning_rate}")


class HybridActOutput(NamedTuple):
    """Что возвращает :meth:`REINFORCE.act`.

    Attributes:
        env_action: то, что уходит в среду: сначала непрерывная часть
            в диапазоне [−1, 1], затем индексы дискретных веток.
        raw_continuous: сэмпл непрерывной части **до** приведения диапазона;
            именно для него считается log π.
        discrete: индексы выбранных веток, ``(B, num_branches)``.
        log_prob: log π(a|s) на момент сбора, ``(B,)``.
    """

    env_action: np.ndarray
    raw_continuous: np.ndarray
    discrete: np.ndarray
    log_prob: np.ndarray


@dataclass
class EpisodeBatch:
    """Собранные завершённые эпизоды, вытянутые в один батч.

    Attributes:
        obs: список массивов по сенсорам, каждый ``(N, *shape)``.
        continuous: сырые непрерывные действия ``(N, continuous_dim)``.
        discrete: индексы веток ``(N, num_branches)``.
        returns: фактические возвраты ``G_t``, ``(N,)``.
    """

    obs: list[np.ndarray]
    continuous: np.ndarray
    discrete: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return int(self.returns.shape[0])


class REINFORCE:
    """Monte-Carlo policy gradient с baseline, для гибридных действий.

    Args:
        policy_net: политика, возвращающая пару ``(среднее, логиты)`` и
            умеющая ``log_prob`` и ``entropy`` (см.
            :class:`labrl.nets.hybrid_policy.GridHybridPolicy`).
        value_net: baseline ``V(s)``. ``None`` — обучение без baseline,
            то есть «чистый» REINFORCE из урока 2.1.
        cfg: гиперпараметры.
        device: устройство вычислений.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module | None = None,
        cfg: ReinforceConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or ReinforceConfig()
        self.device = torch.device(device)

        self.policy_net = policy_net.to(self.device)
        self.value_net = value_net.to(self.device) if value_net is not None else None

        parameters = list(self.policy_net.parameters())
        if self.value_net is not None:
            parameters += list(self.value_net.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.cfg.learning_rate)
        self.updates = 0

    @property
    def continuous_dim(self) -> int:
        return int(self.policy_net.continuous_dim)

    @property
    def discrete_branches(self) -> tuple[int, ...]:
        return tuple(self.policy_net.discrete_branches)

    # --- взаимодействие со средой ---------------------------------------

    def _tensors(self, obs: Sequence[np.ndarray]) -> list[torch.Tensor]:
        return [torch.as_tensor(np.asarray(o, dtype=np.float32), device=self.device) for o in obs]

    @torch.no_grad()
    def act(self, obs: Sequence[np.ndarray], rng: np.random.Generator) -> HybridActOutput:
        """Сэмплирует гибридное действие для батча наблюдений.

        Случайность берётся из переданного генератора numpy — и для гауссова
        шума, и для выбора ветки: воспроизводимость прогона не должна зависеть
        от глобального состояния torch.
        """
        tensors = self._tensors(obs)
        mean, logits = self.policy_net(*tensors)
        std = self.policy_net.clamped_log_std().exp()

        noise = torch.as_tensor(
            rng.standard_normal(size=tuple(mean.shape)).astype(np.float32), device=self.device)
        raw = mean + std * noise

        discrete_columns = []
        for branch_logits in self.policy_net.split_logits(logits):
            probabilities = torch.softmax(branch_logits, dim=1).cpu().numpy()
            # Обратное преобразование выборки: один общий вызов rng на батч
            # вместо цикла по строкам.
            cumulative = probabilities.cumsum(axis=1)
            draws = rng.random(size=(probabilities.shape[0], 1))
            discrete_columns.append((draws < cumulative).argmax(axis=1))
        discrete = np.stack(discrete_columns, axis=1).astype(np.int64)

        log_prob = self.policy_net.log_prob(
            *tensors, raw, torch.as_tensor(discrete, device=self.device))

        raw_np = raw.cpu().numpy()
        env_action = np.concatenate(
            [to_env_action(raw_np), discrete.astype(np.float32)], axis=1)

        return HybridActOutput(
            env_action=env_action.astype(np.float32),
            raw_continuous=raw_np,
            discrete=discrete,
            log_prob=log_prob.cpu().numpy(),
        )

    @torch.no_grad()
    def deterministic_action(self, obs: Sequence[np.ndarray]) -> np.ndarray:
        """Детерминированное действие: среднее непрерывной части и argmax веток.

        Это **в точности** то, что вычисляет экспортированный граф ONNX
        (`deterministic_continuous_actions` и `deterministic_discrete_actions`),
        поэтому оценка в Python и поведение в Unity сравнимы напрямую.
        """
        tensors = self._tensors(obs)
        mean, logits = self.policy_net(*tensors)
        continuous = to_env_action(mean.cpu().numpy())
        discrete = np.stack(
            [branch.argmax(dim=1).cpu().numpy()
             for branch in self.policy_net.split_logits(logits)], axis=1)
        return np.concatenate([continuous, discrete.astype(np.float32)], axis=1).astype(np.float32)

    @torch.no_grad()
    def baseline(self, obs: Sequence[np.ndarray]) -> np.ndarray:
        """``V(s)`` baseline. Нули, если baseline отключён."""
        if self.value_net is None:
            return np.zeros(len(obs[0]), dtype=np.float32)
        return self.value_net(*self._tensors(obs)).squeeze(-1).cpu().numpy()

    def set_learning_rate(self, learning_rate: float) -> None:
        """Меняет шаг Adam — для расписания."""
        if learning_rate < 0.0:
            raise ValueError(f"learning_rate должен быть >= 0, получено {learning_rate}")
        self.cfg.learning_rate = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    # --- обучение --------------------------------------------------------

    def update(self, batch: EpisodeBatch) -> dict[str, float]:
        """Единственное место, где меняются параметры (требование 8.7).

        Один шаг градиента по всем собранным эпизодам. Данные после этого
        выбрасываются: как и у любого on-policy метода, они описывали
        политику, которой больше нет.
        """
        obs = self._tensors(batch.obs)
        continuous = torch.as_tensor(batch.continuous, dtype=torch.float32, device=self.device)
        discrete = torch.as_tensor(batch.discrete, dtype=torch.int64, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)

        if self.value_net is not None:
            values = self.value_net(*obs).squeeze(-1)
            advantage = returns - values.detach()
        else:
            values = None
            advantage = returns

        if self.cfg.normalize_advantage and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        log_prob = self.policy_net.log_prob(*obs, continuous, discrete)
        entropy = self.policy_net.entropy(*obs)

        # Знак минус: оптимизатор минимизирует, а policy gradient — подъём.
        policy_loss = -(log_prob * advantage).mean()
        value_loss = (F.mse_loss(values, returns) if values is not None
                      else torch.zeros((), device=self.device))
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(self.policy_net.parameters())
        if self.value_net is not None:
            parameters += list(self.value_net.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf"))
        self.optimizer.step()
        self.updates += 1

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": float(grad_norm),
            "return_mean": float(returns.mean().item()),
            "return_std": float(returns.std().item()) if returns.numel() > 1 else 0.0,
            "log_std": float(self.policy_net.clamped_log_std().mean().item()),
            "updates": float(self.updates),
        }

    # --- сохранение и экспорт -------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }
        if self.value_net is not None:
            state["value_net"] = self.value_net.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(state["policy_net"])
        if self.value_net is not None and "value_net" in state:
            self.value_net.load_state_dict(state["value_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))

    def policy_module(self) -> nn.Module:
        """Модуль, экспортируемый в ONNX: сама политика.

        Она возвращает пару ``(среднее, логиты)`` — формат гибридного
        контракта. Baseline в Unity не нужен: он существовал только ради
        снижения дисперсии при обучении.
        """
        return self.policy_net.eval()


def discounted_returns(rewards: Sequence[float], gamma: float,
                       bootstrap: float = 0.0) -> np.ndarray:
    """Фактические возвраты ``G_t`` по эпизоду, считая с конца.

    Args:
        rewards: награды эпизода по шагам.
        gamma: коэффициент дисконтирования.
        bootstrap: ценность состояния **после** последнего шага. Ноль для
            истинного завершения; для эпизода, оборванного по ``MaxStep``,
            сюда подставляется оценка baseline — иначе метод выучит, что
            нехватка времени равносильна провалу.

    Returns:
        ``(T,)`` возвраты в том же порядке, что и награды.
    """
    result = np.zeros(len(rewards), dtype=np.float64)
    running = float(bootstrap)
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        result[t] = running
    return result
