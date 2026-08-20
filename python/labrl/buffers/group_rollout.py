"""Буфер роллаута для команды агентов (MA-POCA).

Чем он отличается от :mod:`labrl.buffers.rollout`. Обычный роллаут хранит
переход **одного** агента: наблюдение, действие, награда. Здесь единица
хранения — **шаг команды**: наблюдения всех её игроков, действия всех
её игроков и одна общая награда.

Почему награда общая. Гол засчитан команде: Unity начисляет его через
``SimpleMultiAgentGroup.AddGroupReward``, и низкоуровневый API отдаёт его
в поле ``group_reward``, одинаковом у всех членов группы. Личная награда
у игрока тоже есть (плата за время), и она складывается с командной::

    r_команды(t) = r_группы(t) + Σ_i r_i(t)

Эта сумма и есть то, что предсказывает централизованный критик V. Разложить
её обратно по игрокам буфер не пытается — это работа контрфактического
базлайна в :mod:`labrl.algos.mapoca`, и именно в этом состоит задача
распределения заслуги.

Что буфер считает, а что нет
----------------------------
Считает **λ-возврат** ``G^λ_t`` по траектории значений критика — цель
и для V, и для базлайна. Преимущество ``A_i = G^λ − Q_i`` не считает:
``Q_i`` зависит от действий остальных игроков и вычисляется сетью уже
на этапе обновления, по всему батчу сразу.

Обрыв по времени различается так же, как в одноагентном случае
(требование 8.3): ``EndGroupEpisode`` в Unity — истинное завершение,
``GroupEpisodeInterrupted`` — обрыв, при котором ценность будущего
обязана подставляться из критика.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GroupRolloutBatch:
    """Готовый к обучению роллаут команды. ``M`` — число шагов команды.

    Attributes:
        obs: ``(M, n, obs_dim)`` — наблюдения всех игроков команды.
        action: ``(M, n, num_branches)`` — их действия, индексы веток.
        active: ``(M, n)`` — 1 у игрока, присутствовавшего на шаге, иначе 0.
        log_prob: ``(M, n)`` — ``log π_старой(a_i|o_i)`` на момент сбора.
        returns: ``(M,)`` — λ-возврат команды; цель и для V, и для базлайна.
        value: ``(M,)`` — ``V`` на момент сбора.
    """

    obs: np.ndarray
    action: np.ndarray
    active: np.ndarray
    log_prob: np.ndarray
    returns: np.ndarray
    value: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    @property
    def team_size(self) -> int:
        return int(self.obs.shape[1])


@dataclass
class _GroupStep:
    """Один шаг команды."""

    obs: np.ndarray          # (n, obs_dim)
    action: np.ndarray       # (n, num_branches)
    active: np.ndarray       # (n,)
    log_prob: np.ndarray     # (n,)
    value: float
    reward: float
    terminated: bool
    truncated: bool
    #: ``V`` последнего состояния оборванного эпизода. Осмысленно при truncated.
    bootstrap_value: float = 0.0


@dataclass
class GroupRolloutBuffer:
    """Накопитель роллаута по командам.

    Args:
        num_groups: сколько команд собирается параллельно (по одной на арену).
        gamma: коэффициент дисконтирования.
        gae_lambda: параметр λ. При λ = 1 λ-возврат вырождается в полную
            дисконтированную сумму наград, при λ = 0 — в одношаговую цель
            ``r + γV(s′)``.
    """

    num_groups: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    _groups: list[list[_GroupStep]] = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda должна быть в [0, 1], получено {self.gae_lambda}")
        self._groups = [[] for _ in range(int(self.num_groups))]

    def __len__(self) -> int:
        return sum(len(steps) for steps in self._groups)

    def clear(self) -> None:
        self._groups = [[] for _ in range(int(self.num_groups))]

    def add(
        self,
        group: int,
        obs: np.ndarray,
        action: np.ndarray,
        active: np.ndarray,
        log_prob: np.ndarray,
        value: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Добавляет один шаг команды."""
        self._groups[group].append(
            _GroupStep(
                obs=np.asarray(obs, dtype=np.float32).copy(),
                action=np.asarray(action, dtype=np.int64).copy(),
                active=np.asarray(active, dtype=np.float32).copy(),
                log_prob=np.asarray(log_prob, dtype=np.float32).copy(),
                value=float(value),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                bootstrap_value=float(bootstrap_value),
            )
        )

    def compute(self, last_values: np.ndarray) -> GroupRolloutBatch:
        """Считает λ-возвраты и возвращает плоский батч.

        Args:
            last_values: ``(num_groups,)`` — ``V`` текущего состояния команды.
                Продолжение цепочки для команды, чей последний записанный шаг
                эпизод не завершил.
        """
        last_values = np.asarray(last_values, dtype=np.float64)
        if last_values.shape != (self.num_groups,):
            raise ValueError(
                f"last_values должен быть формы {(self.num_groups,)}, получено {last_values.shape}"
            )

        obs_out, act_out, active_out, logp_out, ret_out, val_out = [], [], [], [], [], []

        for group, steps in enumerate(self._groups):
            if not steps:
                continue

            returns = np.zeros(len(steps), dtype=np.float64)
            advantage = 0.0
            next_value = float(last_values[group])

            for t in reversed(range(len(steps))):
                step = steps[t]
                if step.terminated or step.truncated:
                    # Истинное завершение обнуляет будущее; обрыв по времени —
                    # нет, там ценность подставляется из критика (8.3).
                    bootstrap = 0.0 if step.terminated else step.bootstrap_value
                    delta = step.reward + self.gamma * bootstrap - step.value
                    # Цепочка прерывается: следующий шаг команды — другой матч.
                    advantage = delta
                else:
                    delta = step.reward + self.gamma * next_value - step.value
                    advantage = delta + self.gamma * self.gae_lambda * advantage

                # λ-возврат: A^GAE + V — это в точности TD(λ)-цель.
                returns[t] = advantage + step.value
                next_value = step.value

            obs_out.append(np.stack([s.obs for s in steps]))
            act_out.append(np.stack([s.action for s in steps]))
            active_out.append(np.stack([s.active for s in steps]))
            logp_out.append(np.stack([s.log_prob for s in steps]))
            ret_out.append(returns)
            val_out.append(np.array([s.value for s in steps], dtype=np.float64))

        if not obs_out:
            raise RuntimeError("роллаут пуст: ни одна команда не дала ни одного шага")

        return GroupRolloutBatch(
            obs=np.concatenate(obs_out).astype(np.float32),
            action=np.concatenate(act_out).astype(np.int64),
            active=np.concatenate(active_out).astype(np.float32),
            log_prob=np.concatenate(logp_out).astype(np.float32),
            returns=np.concatenate(ret_out).astype(np.float32),
            value=np.concatenate(val_out).astype(np.float32),
        )
