"""Расписания гиперпараметров (ε, learning rate, β и т.п.).

Расписание — функция «номер шага сбора опыта -> значение». Все значения,
влияющие на результат, задаются конфигом, а не константами внутри алгоритма
(требование 8.6), поэтому расписание передаётся в алгоритм объектом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Schedule(Protocol):
    """Минимальный протокол расписания."""

    def __call__(self, step: int) -> float:  # pragma: no cover - протокол
        ...


@dataclass(frozen=True)
class ConstantSchedule:
    """Постоянное значение."""

    value: float

    def __call__(self, step: int) -> float:
        return self.value


@dataclass(frozen=True)
class LinearSchedule:
    """Линейная интерполяция ``start -> end`` за ``decay_steps`` шагов.

    После ``decay_steps`` значение остаётся равным ``end``.

    Args:
        start: значение на шаге 0.
        end: значение на шаге ``decay_steps`` и далее.
        decay_steps: длительность спада в шагах сбора опыта (steps).
    """

    start: float
    end: float
    decay_steps: int

    def __post_init__(self) -> None:
        if self.decay_steps <= 0:
            raise ValueError(f"decay_steps должен быть > 0, получено {self.decay_steps}")

    def __call__(self, step: int) -> float:
        frac = min(max(step, 0) / self.decay_steps, 1.0)
        return self.start + frac * (self.end - self.start)


@dataclass(frozen=True)
class ExponentialSchedule:
    """Экспоненциальный спад с полом: ``max(end, start * rate ** step)``."""

    start: float
    end: float
    rate: float

    def __post_init__(self) -> None:
        if not 0.0 < self.rate < 1.0:
            raise ValueError(f"rate должен быть в (0, 1), получено {self.rate}")

    def __call__(self, step: int) -> float:
        return max(self.end, self.start * (self.rate ** max(step, 0)))


def build_schedule(spec: dict | float | int) -> Schedule:
    """Собирает расписание из блока конфига **или из числа**.

    Ожидаемые формы блока::

        {"type": "constant", "value": 0.1}
        {"type": "linear", "start": 1.0, "end": 0.05, "decay_steps": 100000}
        {"type": "exponential", "start": 1.0, "end": 0.05, "rate": 0.9999}

    Число трактуется как постоянное расписание: `docs/03_CONVENTIONS.md`, §6
    разрешает обе записи, и требовать блок ради ``epsilon: 0.0`` в методе,
    который ε вообще не использует, — лишняя церемония. Без этой ветки
    конфиг с числом падал с ``TypeError: 'float' object is not subscriptable``
    уже после открытия среды.
    """
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return ConstantSchedule(float(spec))
    if not isinstance(spec, dict):
        raise ValueError(
            f"расписание задаётся числом или блоком с ключом type, получено {type(spec).__name__}"
        )

    kind = spec["type"]
    args = {k: v for k, v in spec.items() if k != "type"}
    if kind == "constant":
        return ConstantSchedule(**args)
    if kind == "linear":
        return LinearSchedule(**args)
    if kind == "exponential":
        return ExponentialSchedule(**args)
    raise ValueError(f"неизвестный тип расписания: {kind!r}")
