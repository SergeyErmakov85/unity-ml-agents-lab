"""Value Iteration — динамическое программирование при известной модели.

Отличие от Q-learning в одной фразе: здесь **не нужен опыт**, нужна модель.
Алгоритм не взаимодействует со средой вообще — он итеративно применяет
оператор оптимальности Беллмана к таблице значений::

    V_{k+1}(s) = max_a  Σ_{s'} P(s'|s,a) · [ R(s,a,s') + γ · V_k(s') ]

Для детерминированной среды сумма схлопывается в одно слагаемое::

    V_{k+1}(s) = max_a  [ R(s,a) + γ · (1 − terminated(s,a)) · V_k(s'(s,a)) ]

Оператор является сжатием с коэффициентом γ, поэтому последовательность
сходится к единственной неподвижной точке `V*` из любого начального
приближения. Критерий остановки — максимальное изменение значения за итерацию
(`sup`-норма) меньше порога `theta`.

Полученный `Q*` — эталон, с которым сравнивается результат Q-learning: если
жадные политики совпали, обучение из опыта нашло оптимум.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from labrl.envs.gridworld_mdp import GridWorldMDP


@dataclass
class ValueIterationConfig:
    """Гиперпараметры метода."""

    #: Коэффициент дисконтирования, [0, 1). Он же определяет скорость сходимости.
    gamma: float = 0.99
    #: Порог остановки по sup-норме изменения V за итерацию.
    theta: float = 1e-10
    #: Предохранитель от бесконечного цикла при некорректной модели.
    max_iterations: int = 10_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if self.theta <= 0.0:
            raise ValueError(f"theta должна быть > 0, получено {self.theta}")


@dataclass
class ValueIterationResult:
    """Результат: значения, Q-функция, политика и история сходимости."""

    values: np.ndarray        # (num_states,)
    q: np.ndarray             # (num_states, num_actions)
    policy: np.ndarray        # (num_states,)
    iterations: int
    deltas: list[float]       # sup-норма изменения V по итерациям

    @property
    def converged(self) -> bool:
        return len(self.deltas) < 1 or self.deltas[-1] <= 0.0 or self.iterations < 10_000


class ValueIteration:
    """Value Iteration на явной модели MDP.

    Args:
        cfg: гиперпараметры.
    """

    def __init__(self, cfg: ValueIterationConfig | None = None) -> None:
        self.cfg = cfg or ValueIterationConfig()

    def solve(self, mdp: GridWorldMDP) -> ValueIterationResult:
        """Находит `V*`, `Q*` и оптимальную политику.

        Returns:
            :class:`ValueIterationResult`.

        Raises:
            RuntimeError: не сошлось за ``max_iterations`` — признак ошибки
                в модели (например, γ ≥ 1 или недостижимые терминальные состояния).
        """
        next_state, reward, terminated = mdp.transition_tables()
        values = np.zeros(mdp.num_states, dtype=np.float64)
        deltas: list[float] = []

        for iteration in range(1, self.cfg.max_iterations + 1):
            # Будущее обнуляется только на терминальных переходах — то же
            # правило, что и в Q-learning (см. labrl.algos.tabular.q_learning).
            future = np.where(terminated, 0.0, values[next_state])
            q = reward + self.cfg.gamma * future
            updated = q.max(axis=1)

            delta = float(np.max(np.abs(updated - values)))
            deltas.append(delta)
            values = updated

            if delta < self.cfg.theta:
                return ValueIterationResult(
                    values=values,
                    q=q,
                    policy=np.argmax(q, axis=1),
                    iterations=iteration,
                    deltas=deltas,
                )

        raise RuntimeError(
            f"Value Iteration не сошлась за {self.cfg.max_iterations} итераций "
            f"(последняя δ = {deltas[-1]:.3e}, порог {self.cfg.theta:.1e}). "
            "Проверьте модель MDP: γ < 1 и достижимость терминальных состояний."
        )


def optimal_action_mask(q_star: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """``(num_states, num_actions)`` — какие действия оптимальны в каждом состоянии.

    В сетке с симметриями оптимальных действий обычно несколько: из клетки
    можно пойти на север или на восток и прийти к цели за одно и то же число
    шагов. Поэтому «правильность» выученной политики нельзя проверять
    совпадением с одной конкретной оптимальной политикой.
    """
    q = np.asarray(q_star, dtype=np.float64)
    return q >= (q.max(axis=1, keepdims=True) - tol)


def policy_is_optimal(
    policy: np.ndarray,
    q_star: np.ndarray,
    mask: np.ndarray | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """``(num_states,)`` bool — выбрала ли политика оптимальное действие.

    Это и есть корректный критерий «политика оптимальна»: действие обязано
    достигать ``max_a Q*(s, a)``, а не совпадать с чьим-то конкретным argmax.

    Args:
        policy: ``(num_states,)`` индексы действий.
        q_star: ``(num_states, num_actions)`` — точная Q-функция из DP.
        mask: какие состояния учитывать; вне маски результат — ``True``.
        tol: допуск на равенство значений.
    """
    policy = np.asarray(policy, dtype=np.int64)
    is_optimal = optimal_action_mask(q_star, tol)[np.arange(len(policy)), policy]
    if mask is None:
        return is_optimal
    return is_optimal | ~np.asarray(mask, dtype=bool)


def optimality_rate(
    policy: np.ndarray,
    q_star: np.ndarray,
    mask: np.ndarray | None = None,
    tol: float = 1e-8,
) -> float:
    """Доля состояний (в пределах маски), где политика выбрала оптимальное действие."""
    optimal = optimal_action_mask(q_star, tol)[np.arange(len(policy)), np.asarray(policy, dtype=np.int64)]
    if mask is None:
        return float(np.mean(optimal))
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("маска не выделяет ни одного состояния")
    return float(np.mean(optimal[mask]))


def policy_agreement(policy_a: np.ndarray, policy_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Доля состояний, где две политики выбирают одно и то же действие.

    Args:
        policy_a, policy_b: ``(num_states,)``.
        mask: какие состояния учитывать. Терминальные и стены обычно исключают:
            действие в них ни на что не влияет, и расхождение там не значит ничего.
    """
    a = np.asarray(policy_a)
    b = np.asarray(policy_b)
    if a.shape != b.shape:
        raise ValueError(f"формы политик не совпадают: {a.shape} и {b.shape}")
    if mask is None:
        return float(np.mean(a == b))
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("маска не выделяет ни одного состояния")
    return float(np.mean(a[mask] == b[mask]))
