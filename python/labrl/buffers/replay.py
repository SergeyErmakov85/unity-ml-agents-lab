"""Буфер воспроизведения (experience replay) — равномерная выборка.

Зачем он нужен. Последовательные переходы в RL сильно коррелированы: соседние
шаги почти одинаковы. Обучать сеть на такой выборке — то же, что обучать
классификатор на отсортированном по классам датасете: градиенты «тянут»
в одну сторону, и сеть забывает то, что видела раньше. Буфер перемешивает
опыт во времени и превращает поток коррелированных переходов в приближённо
независимую выборку.

Второе свойство, не менее важное: каждый переход используется многократно.
Шаг в среде Unity стоит на порядки дороже, чем шаг оптимизатора, и
переиспользование опыта — главный источник эффективности value-based методов.

Реализация — кольцевой буфер на заранее выделенных массивах numpy. Списки
и `deque` здесь не годятся: при миллионе переходов пересборка батча из
питоновских объектов стоит дороже, чем сам шаг обучения.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    """Батч переходов. Все поля — массивы длины ``batch_size``."""

    obs: np.ndarray          # (B, obs_dim)
    action: np.ndarray       # (B,) int64
    reward: np.ndarray       # (B,) float32
    next_obs: np.ndarray     # (B, obs_dim)
    terminated: np.ndarray   # (B,) bool — истинное завершение
    truncated: np.ndarray    # (B,) bool — обрыв по MaxStep

    def __len__(self) -> int:
        return self.action.shape[0]


class ReplayBuffer:
    """Кольцевой буфер переходов с равномерной выборкой.

    Args:
        capacity: сколько переходов хранить. При переполнении затираются
            самые старые.
        obs_dim: размерность наблюдения.
        seed: сид генератора выборки — прогон должен воспроизводиться.

    Про хранение ``terminated`` и ``truncated`` **раздельно**: цель обновления
    обнуляет будущее только при ``terminated``. Если сохранить один флаг
    ``done``, различие восстановить уже невозможно, и обучение будет тихо
    искажено (см. `labrl.algos.dqn`).
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity должна быть > 0, получено {capacity}")

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._rng = np.random.default_rng(seed)

        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._action = np.zeros(self.capacity, dtype=np.int64)
        self._reward = np.zeros(self.capacity, dtype=np.float32)
        self._terminated = np.zeros(self.capacity, dtype=bool)
        self._truncated = np.zeros(self.capacity, dtype=bool)

        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> int:
        """Добавляет пачку переходов (по одному с каждой активной арены).

        Returns:
            Сколько переходов добавлено.
        """
        n = len(action)
        if n == 0:
            return 0
        if obs.shape != (n, self.obs_dim):
            raise ValueError(f"obs должен иметь форму {(n, self.obs_dim)}, получено {obs.shape}")

        # Запись может пересечь конец кольца — тогда она делится на два куска.
        idx = (self._cursor + np.arange(n)) % self.capacity
        self._obs[idx] = obs
        self._next_obs[idx] = next_obs
        self._action[idx] = action
        self._reward[idx] = reward
        self._terminated[idx] = terminated
        self._truncated[idx] = truncated

        self._cursor = int((self._cursor + n) % self.capacity)
        self._size = int(min(self._size + n, self.capacity))
        return n

    def sample(self, batch_size: int) -> Batch:
        """Равномерная выборка **с возвращением**.

        С возвращением — потому что так делает исходный DQN и потому что при
        буфере в сотни тысяч переходов вероятность повтора внутри батча
        пренебрежимо мала, а код проще и быстрее.
        """
        if self._size == 0:
            raise RuntimeError("буфер пуст: нечего выбирать")
        if batch_size <= 0:
            raise ValueError(f"batch_size должен быть > 0, получено {batch_size}")

        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self._obs[idx],
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs=self._next_obs[idx],
            terminated=self._terminated[idx],
            truncated=self._truncated[idx],
        )
