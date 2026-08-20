"""Буфер роллаута для on-policy методов (A2C, PPO) и обобщённое преимущество.

Чем он отличается от буфера воспроизведения. Буфер воспроизведения
(:mod:`labrl.buffers.replay`) хранит опыт **любой** политики и переиспользует
его многократно — это возможно только для off-policy методов. A2C и PPO
обучаются на опыте **текущей** политики: как только параметры изменились,
собранные данные устарели. Поэтому роллаут собирается, один раз используется
и выбрасывается.

Обобщённое преимущество (GAE, Schulman et al., 2016)
----------------------------------------------------
Преимущество ``A(s, a) = Q(s, a) − V(s)`` отвечает на вопрос «насколько это
действие лучше среднего». Оценить его можно по-разному, и выбор — это выбор
между смещением и дисперсией::

    δ_t     = r_t + γ·V(s_{t+1}) − V(s_t)          одношаговая TD-ошибка
    A^GAE_t = Σ_{l≥0} (γλ)^l · δ_{t+l}

При λ = 0 остаётся δ_t — низкая дисперсия, но смещение от неточного критика.
При λ = 1 получается ``Σ γ^l r − V(s_t)`` — несмещённо, но дисперсия растёт
с длиной эпизода. Практическое λ ≈ 0.95 берёт почти всё преимущество обоих.

Обрыв по времени
----------------
Различие ``terminated`` и ``truncated`` здесь так же принципиально, как
в DQN. При истинном завершении будущего нет: ``V(s_{t+1}) = 0``. При обрыве
по ``MaxStep`` эпизод продолжался бы, и его ценность обязана быть подставлена
из критика — иначе метод учится считать нехватку времени катастрофой.
В обоих случаях **цепочка GAE прерывается**: следующая запись слота относится
уже к другому эпизоду.

Асинхронность слотов
--------------------
Векторизованная среда не гарантирует, что все K слотов дают запись на каждом
шаге (слот, только что завершивший эпизод, может пропустить шаг). Поэтому
буфер хранит **последовательность на слот**, а не прямоугольную матрицу
``(T, K)``: свёртка GAE идёт по фактической истории каждого слота.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBatch:
    """Готовый к обучению роллаут. Все массивы одной длины N."""

    obs: np.ndarray          # (N, obs_dim)
    action: np.ndarray       # (N, action_dim) — сырое действие политики, до клиппинга
    log_prob: np.ndarray     # (N,)  log π_старой(a|s) на момент сбора
    advantage: np.ndarray    # (N,)
    returns: np.ndarray      # (N,)  цель для критика: advantage + V(s)
    value: np.ndarray        # (N,)  V(s) на момент сбора

    def __len__(self) -> int:
        return int(self.obs.shape[0])


@dataclass
class _Step:
    """Одна запись слота. Поля названы как в формулах выше."""

    obs: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    terminated: bool
    truncated: bool
    #: V(s_последнее) для оборванного эпизода. Осмысленно только при truncated.
    bootstrap_value: float = 0.0


@dataclass
class RolloutBuffer:
    """Накопитель роллаута по слотам.

    Args:
        num_envs: число слотов векторизованной среды.
        gamma: коэффициент дисконтирования.
        gae_lambda: параметр λ обобщённого преимущества.
    """

    num_envs: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    _slots: list[list[_Step]] = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda должна быть в [0, 1], получено {self.gae_lambda}")
        self._slots = [[] for _ in range(int(self.num_envs))]

    def __len__(self) -> int:
        return sum(len(steps) for steps in self._slots)

    def clear(self) -> None:
        self._slots = [[] for _ in range(int(self.num_envs))]

    def add(
        self,
        slot: int,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Добавляет один переход слота."""
        self._slots[slot].append(
            _Step(
                obs=np.asarray(obs, dtype=np.float32).copy(),
                action=np.asarray(action, dtype=np.float32).copy(),
                log_prob=float(log_prob),
                value=float(value),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                bootstrap_value=float(bootstrap_value),
            )
        )

    def compute(self, last_values: np.ndarray) -> RolloutBatch:
        """Считает преимущества и цели критика, возвращая плоский батч.

        Args:
            last_values: ``(num_envs,)`` — ``V(s)`` текущего наблюдения слота.
                Используется как продолжение цепочки для слота, чей последний
                записанный шаг не завершил эпизод.
        """
        last_values = np.asarray(last_values, dtype=np.float64)
        if last_values.shape != (self.num_envs,):
            raise ValueError(
                f"last_values должен быть формы {(self.num_envs,)}, получено {last_values.shape}"
            )

        obs_out, act_out = [], []
        logp_out, adv_out, ret_out, val_out = [], [], [], []

        for slot, steps in enumerate(self._slots):
            if not steps:
                continue

            advantages = np.zeros(len(steps), dtype=np.float64)
            running_gae = 0.0
            # Ценность состояния, следующего за последним записанным шагом.
            next_value = float(last_values[slot])

            for t in reversed(range(len(steps))):
                step = steps[t]
                if step.terminated or step.truncated:
                    # Истинное завершение обнуляет будущее; обрыв по времени —
                    # нет, там ценность подставляется из критика.
                    bootstrap = 0.0 if step.terminated else step.bootstrap_value
                    delta = step.reward + self.gamma * bootstrap - step.value
                    # Цепочка прерывается: следующий шаг слота — другой эпизод.
                    running_gae = delta
                else:
                    delta = step.reward + self.gamma * next_value - step.value
                    running_gae = delta + self.gamma * self.gae_lambda * running_gae

                advantages[t] = running_gae
                next_value = step.value

            values = np.array([s.value for s in steps], dtype=np.float64)

            obs_out.append(np.stack([s.obs for s in steps]))
            act_out.append(np.stack([s.action for s in steps]))
            logp_out.append(np.array([s.log_prob for s in steps], dtype=np.float64))
            adv_out.append(advantages)
            # Цель критика — то же, что и в TD(λ): A + V. Отдельно считать
            # дисконтированную сумму наград не нужно, это она и есть.
            ret_out.append(advantages + values)
            val_out.append(values)

        if not obs_out:
            raise RuntimeError("роллаут пуст: ни один слот не дал ни одного перехода")

        return RolloutBatch(
            obs=np.concatenate(obs_out).astype(np.float32),
            action=np.concatenate(act_out).astype(np.float32),
            log_prob=np.concatenate(logp_out).astype(np.float32),
            advantage=np.concatenate(adv_out).astype(np.float32),
            returns=np.concatenate(ret_out).astype(np.float32),
            value=np.concatenate(val_out).astype(np.float32),
        )
