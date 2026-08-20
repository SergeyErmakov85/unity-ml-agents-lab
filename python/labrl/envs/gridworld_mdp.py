"""Явная модель MDP для `E01_GridWorld` — реплика среды Unity на стороне Python.

Зачем она нужна. Динамическое программирование (Value Iteration, Policy
Iteration) требует **знания модели**: вероятностей переходов и наград. Q-learning
модели не требует. Иметь обе реализации на одной среде — и есть смысл примера:
Value Iteration даёт точный оптимум `Q*`, с которым сравнивается то, чему
Q-learning научился из опыта.

Вторая, не менее важная роль — **проверка корректности MDP**. Реплика написана
по тем же правилам, что и `GridWorldEnvironment.cs`, но независимо. Если прогон
реальной среды Unity расходится с предсказанием реплики хотя бы на одном
переходе, значит одна из двух реализаций неверна — и это обнаруживается сразу,
а не через часы обучения. Сверка выполняется функцией :func:`check_against_env`.

Соглашения совпадают с `GridWorldEnvironment.cs`:

* клетка — ``(c, r)``, индекс состояния ``s = r · cols + c``;
* действия ``0=N (+r), 1=S (−r), 2=E (+c), 3=W (−c)``;
* выход за границу или в стену — ход блокирован, агент остаётся на месте.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Смещения (dc, dr) для действий 0..3 — порядок как в C#-коде среды.
DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))

#: Метки действий для читаемых распечаток политики.
ACTION_LABELS: tuple[str, ...] = ("N", "S", "E", "W")


@dataclass(frozen=True)
class GridWorldMDP:
    """Модель среды `E01_GridWorld`. Значения по умолчанию — из TS-001.

    Attributes:
        cols, rows: размер сетки.
        start: стартовая клетка ``(c, r)``.
        goal: целевая клетка.
        traps: клетки-ловушки.
        walls: непроходимые клетки.
        step_reward: награда за каждый шаг.
        goal_reward: дополнительная награда за достижение цели.
        trap_reward: дополнительная награда (штраф) за ловушку.
        max_steps: бюджет шагов эпизода (``MaxStep`` агента).
    """

    cols: int = 5
    rows: int = 5
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)
    traps: tuple[tuple[int, int], ...] = ((1, 3), (3, 1))
    walls: tuple[tuple[int, int], ...] = ((1, 2), (3, 3), (3, 4))
    step_reward: float = -0.04
    goal_reward: float = 1.0
    trap_reward: float = -1.0
    max_steps: int = 100

    num_actions: int = field(default=4, init=False)

    # --- индексация ------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self.cols * self.rows

    def index(self, cell: tuple[int, int]) -> int:
        c, r = cell
        return r * self.cols + c

    def cell(self, state: int) -> tuple[int, int]:
        return (state % self.cols, state // self.cols)

    # --- предикаты клеток ------------------------------------------------

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        c, r = cell
        return 0 <= c < self.cols and 0 <= r < self.rows

    def is_wall(self, cell: tuple[int, int]) -> bool:
        return cell in self.walls

    def is_trap(self, cell: tuple[int, int]) -> bool:
        return cell in self.traps

    def is_goal(self, cell: tuple[int, int]) -> bool:
        return cell == self.goal

    def is_terminal(self, state: int) -> bool:
        """Терминальные состояния — цель и ловушки. Стены недостижимы."""
        cell = self.cell(state)
        return self.is_goal(cell) or self.is_trap(cell)

    # --- динамика --------------------------------------------------------

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        """Один переход детерминированной среды.

        Returns:
            ``(next_state, reward, terminated)``. ``terminated`` — истинное
            завершение (цель или ловушка), обрыв по ``max_steps`` сюда не входит:
            он относится к эпизоду, а не к переходу.
        """
        cell = self.cell(state)
        dc, dr = DIRECTIONS[action]
        candidate = (cell[0] + dc, cell[1] + dr)
        nxt = cell if (not self.in_bounds(candidate) or self.is_wall(candidate)) else candidate

        reward = self.step_reward
        terminated = False
        if self.is_goal(nxt):
            reward += self.goal_reward
            terminated = True
        elif self.is_trap(nxt):
            reward += self.trap_reward
            terminated = True

        return self.index(nxt), reward, terminated

    def transition_tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Полная модель в табличном виде — вход для динамического программирования.

        Returns:
            ``(next_state, reward, terminated)``, каждый формы
            ``(num_states, num_actions)``.
        """
        n, a = self.num_states, self.num_actions
        next_state = np.zeros((n, a), dtype=np.int64)
        reward = np.zeros((n, a), dtype=np.float64)
        terminated = np.zeros((n, a), dtype=bool)

        for s in range(n):
            for act in range(a):
                if self.is_terminal(s):
                    # Из терминального состояния эпизод уже не продолжается;
                    # петля с нулевой наградой оставляет V(терминал) = 0.
                    next_state[s, act], reward[s, act], terminated[s, act] = s, 0.0, True
                    continue
                next_state[s, act], reward[s, act], terminated[s, act] = self.step(s, act)

        return next_state, reward, terminated

    # --- вспомогательное -------------------------------------------------

    def one_hot(self, state: int) -> np.ndarray:
        obs = np.zeros(self.num_states, dtype=np.float32)
        obs[state] = 1.0
        return obs

    def render_policy(self, policy: np.ndarray) -> str:
        """Печатает политику сеткой — «глазами» проверить результат быстрее, чем числами."""
        lines = []
        for r in range(self.rows - 1, -1, -1):
            row = []
            for c in range(self.cols):
                cell = (c, r)
                if self.is_wall(cell):
                    row.append(" # ")
                elif self.is_goal(cell):
                    row.append(" G ")
                elif self.is_trap(cell):
                    row.append(" X ")
                else:
                    row.append(f" {ACTION_LABELS[int(policy[self.index(cell)])]} ")
            lines.append("".join(row))
        return "\n".join(lines)

    def render_values(self, values: np.ndarray) -> str:
        """Печатает V(s) сеткой."""
        lines = []
        for r in range(self.rows - 1, -1, -1):
            row = []
            for c in range(self.cols):
                cell = (c, r)
                row.append("  #### " if self.is_wall(cell) else f"{values[self.index(cell)]:7.3f}")
            lines.append(" ".join(row))
        return "\n".join(lines)


def check_against_env(
    mdp: GridWorldMDP,
    observed: list[tuple[int, int, int, float, bool]],
) -> list[str]:
    """Сверяет наблюдённые в Unity переходы с предсказанием модели.

    Args:
        mdp: модель.
        observed: список ``(state, action, next_state, reward, terminated)``,
            собранный прогоном реальной среды.

    Returns:
        Список расхождений; пустой список означает, что реплика точна.
        Каждая строка называет конкретный переход — расследовать по ней можно
        сразу, без повторного прогона.
    """
    problems: list[str] = []
    for state, action, next_state, reward, terminated in observed:
        exp_next, exp_reward, exp_term = mdp.step(state, action)
        if exp_next != next_state:
            problems.append(
                f"s={state}{mdp.cell(state)} a={ACTION_LABELS[action]}: "
                f"Unity дал s'={next_state}{mdp.cell(next_state)}, модель — {exp_next}{mdp.cell(exp_next)}"
            )
        if abs(exp_reward - reward) > 1e-5:
            problems.append(
                f"s={state} a={ACTION_LABELS[action]}: награда Unity {reward:.4f}, модель {exp_reward:.4f}"
            )
        if exp_term != terminated:
            problems.append(
                f"s={state} a={ACTION_LABELS[action]}: terminated Unity {terminated}, модель {exp_term}"
            )
    return problems
