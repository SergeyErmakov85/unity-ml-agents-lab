"""Табличный Q-learning (Watkins, 1989) — реализация с нуля.

Метод **off-policy**: политика сбора опыта (ε-жадная) отличается от той,
по которой строится цель обновления (жадная). Именно поэтому в цели стоит
``max_a' Q(s', a')``, а не ``Q(s', a_next)`` — последнее было бы SARSA.

Обновление::

    y     = r + γ · (1 − terminated) · max_a' Q(s', a')
    Q(s,a) ← Q(s,a) + α · (y − Q(s,a))

Про множитель ``(1 − terminated)``. Обнуляется будущее **только** при истинном
завершении эпизода. Обрыв по ``MaxStep`` (``truncated``) — не терминальное
состояние: там будущее есть, и бутстрэппинг обязан выполняться. Смешать эти
случаи — значит научить агента считать нехватку времени катастрофой.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from labrl.envs.state_encoders import StateEncoder


@dataclass
class QLearningConfig:
    """Все гиперпараметры метода. Значений «по месту» в коде нет (требование 8.6)."""

    #: Коэффициент дисконтирования, безразмерный, [0, 1).
    gamma: float = 0.99
    #: Скорость обучения α, безразмерная, (0, 1].
    learning_rate: float = 0.1
    #: Начальное значение Q для всех пар (s, a). Ноль — нейтральная инициализация;
    #: положительное значение даёт «оптимизм в условиях неопределённости»
    #: и само по себе стимулирует разведку.
    initial_q: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(f"learning_rate должен быть в (0, 1], получено {self.learning_rate}")


@dataclass
class Transition:
    """Батч переходов из K параллельных арен.

    Все поля — массивы длины K (или подвыборки K). Батч, а не одиночный переход,
    потому что среда векторизована: K арен дают K переходов за шаг.
    """

    state: np.ndarray        # (B,) индексы состояний
    action: np.ndarray       # (B,) индексы действий
    reward: np.ndarray       # (B,)
    next_state: np.ndarray   # (B,)
    terminated: np.ndarray   # (B,) bool — истинное завершение
    truncated: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # (B,) bool


class QLearning:
    """Табличный Q-learning.

    Args:
        num_states: размер пространства состояний.
        num_actions: число действий (одна дискретная ветка).
        cfg: гиперпараметры.

        encoder: как наблюдение среды превращается в индекс состояния **и как
            то же преобразование попадает в граф ONNX**. По умолчанию —
            :class:`labrl.envs.state_encoders.OneHotStateEncoder`: среда отдаёт
            one-hot вектор состояния (`E01`, `E00`). Для непрерывного
            наблюдения передаётся
            :class:`labrl.envs.state_encoders.BoxDiscretizer` (`E02`).

    Таблица ``Q`` формы ``(num_states, num_actions)`` — единственное состояние
    алгоритма. Нейросети здесь нет: она появляется только на этапе экспорта,
    где таблица оборачивается модулем, который выдаёт кодировщик
    (см. :mod:`labrl.nets.tabular`, :mod:`labrl.nets.discretized`).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        cfg: QLearningConfig | None = None,
        encoder: "StateEncoder | None" = None,
    ) -> None:
        from labrl.envs.state_encoders import OneHotStateEncoder

        self.cfg = cfg or QLearningConfig()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.encoder = encoder if encoder is not None else OneHotStateEncoder(self.num_states)
        if self.encoder.num_states != self.num_states:
            raise ValueError(
                f"кодировщик задаёт {self.encoder.num_states} состояний, "
                f"таблица рассчитана на {self.num_states}"
            )
        self.q = np.full((self.num_states, self.num_actions), self.cfg.initial_q, dtype=np.float64)
        self.updates = 0

    # --- взаимодействие со средой ---------------------------------------

    def act(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        """ε-жадный выбор действия для батча состояний.

        Args:
            state: ``(B,)`` индексы состояний.
            epsilon: вероятность случайного действия.
            rng: генератор — передаётся явно, чтобы прогон был воспроизводим
                без обращения к глобальному состоянию numpy.

        Returns:
            ``(B,)`` индексы действий.
        """
        state = np.asarray(state, dtype=np.int64)
        greedy = self.greedy_action(state)
        if epsilon <= 0.0:
            return greedy
        explore = rng.random(state.shape[0]) < epsilon
        random_actions = rng.integers(0, self.num_actions, size=state.shape[0])
        return np.where(explore, random_actions, greedy)

    def greedy_action(self, state: np.ndarray) -> np.ndarray:
        """Жадное действие: ``argmax_a Q(s, a)``. Это и есть итоговая политика."""
        return np.argmax(self.q[np.asarray(state, dtype=np.int64)], axis=1)

    # --- обучение --------------------------------------------------------

    def update(self, batch: Transition) -> dict[str, float]:
        """Единственное место, где меняется таблица Q (требование 8.7).

        Обработка повторов внутри батча. Если две арены дали переход из одной
        и той же пары (s, a), наивное векторное присваивание применит только
        последнее обновление — остальные потеряются. Поэтому приращения
        суммируются через ``np.add.at``: это эквивалентно последовательному
        применению всех переходов батча с одинаковым «старым» Q, что и есть
        стандартная семантика батчевого табличного обновления.

        Returns:
            Метрики шага обучения для TensorBoard.
        """
        state = np.asarray(batch.state, dtype=np.int64)
        action = np.asarray(batch.action, dtype=np.int64)
        reward = np.asarray(batch.reward, dtype=np.float64)
        next_state = np.asarray(batch.next_state, dtype=np.int64)
        terminated = np.asarray(batch.terminated, dtype=bool)

        if state.size == 0:
            return {"td_error_abs": 0.0, "q_max": float(self.q.max()), "updates": float(self.updates)}

        # Бутстрэппинг: обнуляем будущее только при истинном завершении.
        next_value = self.q[next_state].max(axis=1)
        target = reward + self.cfg.gamma * np.where(terminated, 0.0, next_value)

        current = self.q[state, action]
        td_error = target - current

        np.add.at(self.q, (state, action), self.cfg.learning_rate * td_error)
        self.updates += state.size

        return {
            "td_error_abs": float(np.mean(np.abs(td_error))),
            "td_error_max": float(np.max(np.abs(td_error))),
            "q_max": float(self.q.max()),
            "q_mean": float(self.q.mean()),
            "updates": float(self.updates),
        }

    def set_learning_rate(self, learning_rate: float) -> None:
        """Меняет α — для расписания скорости обучения.

        Зачем α вообще менять. Условие сходимости Роббинса–Монро требует,
        чтобы шаг обучения убывал: сумма шагов расходится, сумма их квадратов
        сходится. При **постоянном** α таблица не сходится, а бесконечно
        колеблется вокруг решения с амплитудой порядка α·|TD-ошибка|. На задачах
        предсказания это почти незаметно, а на задачах управления — фатально:
        колебание Q переворачивает `argmax`, и жадная политика скачет между
        хорошей и негодной от оценки к оценке. Именно это наблюдалось
        в `E02_CartPoleUnity` (см. docs/07_TROUBLESHOOTING.md, T-11).
        """
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(f"learning_rate должен быть в (0, 1], получено {learning_rate}")
        self.cfg.learning_rate = float(learning_rate)

    # --- сохранение и экспорт -------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {"q": self.q.copy(), "updates": self.updates, "cfg": self.cfg}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        q = np.asarray(state["q"], dtype=np.float64)
        if q.shape != (self.num_states, self.num_actions):
            raise ValueError(
                f"форма таблицы {q.shape} не совпадает с ожидаемой "
                f"{(self.num_states, self.num_actions)}"
            )
        self.q = q.copy()
        self.updates = int(state.get("updates", 0))

    def policy_module(self):
        """Модуль, экспортируемый в ONNX. Строится кодировщиком наблюдений.

        При one-hot наблюдении это линейный слой без смещения:
        ``onehot(s) @ Qᵀ == Q[s]``, то есть выход слоя поэлементно равен строке
        таблицы. При непрерывном наблюдении в граф дополнительно попадает сама
        сетка дискретизации — иначе Unity истолковала бы вход иначе, чем
        обучение (требование 10.7).

        В обоих случаях экспортируется **та же самая** политика, а не её
        приближение.
        """
        return self.encoder.policy_module(self.q)

    def greedy_policy(self) -> np.ndarray:
        """``(num_states,)`` — жадное действие в каждом состоянии."""
        return np.argmax(self.q, axis=1)

    def state_values(self) -> np.ndarray:
        """``(num_states,)`` — ``V(s) = max_a Q(s, a)``."""
        return self.q.max(axis=1)
