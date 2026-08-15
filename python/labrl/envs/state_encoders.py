"""Перевод наблюдения среды в индекс состояния таблицы.

Табличные методы (`labrl.algos.tabular`) работают с целым номером состояния,
а среда отдаёт вектор чисел. Кодировщик — это мост между ними, и у него
**две** обязанности, которые нельзя разделять:

1. `index(obs)` — перевод наблюдения в индекс на стороне Python (обучение);
2. `policy_module(q)` — тот же самый перевод, но **внутри графа ONNX**,
   вместе с таблицей.

Почему это один объект, а не два. Если дискретизация живёт только в Python,
экспортированная модель получит на вход сырое наблюдение и истолкует его иначе,
чем обучение, — Unity и Python молча разойдутся. Требование 10.7 запрещает
хранить любое преобразование входа вне графа; здесь этот запрет обеспечен
конструкцией: у кодировщика нельзя взять `index` и «забыть» про `policy_module`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch.nn as nn


class StateEncoder(Protocol):
    """Протокол кодировщика наблюдений для табличных методов."""

    @property
    def num_states(self) -> int:
        """Размер пространства состояний таблицы."""

    def index(self, obs: np.ndarray) -> np.ndarray:
        """``(B, obs_dim) -> (B,)`` индексы состояний."""

    def policy_module(self, q: np.ndarray) -> nn.Module:
        """Экспортируемый модуль: наблюдение -> Q-значения всех действий."""


@dataclass(frozen=True)
class OneHotStateEncoder:
    """Наблюдение уже является one-hot вектором состояния (`E00`, `E01`).

    Перевод в индекс — ``argmax``; обратный перевод в графе не нужен вовсе:
    линейный слой без смещения с весами `Q` при one-hot входе выдаёт строку
    таблицы (см. :class:`labrl.nets.tabular.OneHotQTable`).
    """

    states: int

    @property
    def num_states(self) -> int:
        return int(self.states)

    def index(self, obs: np.ndarray) -> np.ndarray:
        return np.argmax(np.asarray(obs), axis=1).astype(np.int64)

    def policy_module(self, q: np.ndarray) -> nn.Module:
        from labrl.nets.tabular import OneHotQTable

        return OneHotQTable.from_table(q)


@dataclass(frozen=True)
class BoxDiscretizer:
    """Равномерная сетка по непрерывному наблюдению (`E02`).

    Каждое измерение режется границами на ячейки, номер ячейки по всем
    измерениям сворачивается в один индекс (порядок — «последнее измерение
    меняется быстрее», как в C-массивах).

    Границы **разомкнуты слева и справа**: значение ниже первой границы попадает
    в ячейку 0, выше последней — в последнюю ячейку. Поэтому сетку не требуется
    строить по фактическим экстремумам: достаточно покрыть область, где важна
    детализация, а хвосты сольются в крайние ячейки.

    Attributes:
        boundaries: по одному возрастающему массиву границ на измерение.
            Измерение с ``k`` границами даёт ``k + 1`` ячеек.
    """

    boundaries: tuple[tuple[float, ...], ...]

    @classmethod
    def uniform(
        cls,
        lows: Sequence[float],
        highs: Sequence[float],
        bins: Sequence[int],
    ) -> "BoxDiscretizer":
        """Строит сетку с равными ячейками в ``[low, high]`` по каждому измерению.

        Args:
            lows: нижние границы области интереса по измерениям.
            highs: верхние границы.
            bins: число ячеек по измерениям; ``1`` означает «измерение
                игнорируется» (одна ячейка на всю ось).
        """
        if not (len(lows) == len(highs) == len(bins)):
            raise ValueError(
                f"lows, highs и bins должны быть одной длины: {len(lows)}, {len(highs)}, {len(bins)}"
            )

        edges: list[tuple[float, ...]] = []
        for low, high, count in zip(lows, highs, bins):
            if count < 1:
                raise ValueError(f"число ячеек должно быть >= 1, получено {count}")
            if count > 1 and not high > low:
                raise ValueError(f"требуется high > low, получено low={low}, high={high}")
            # count ячеек задаются count-1 внутренними границами.
            inner = np.linspace(low, high, count + 1)[1:-1]
            edges.append(tuple(float(x) for x in inner))
        return cls(boundaries=tuple(edges))

    def __post_init__(self) -> None:
        for dim, edges in enumerate(self.boundaries):
            if any(later <= earlier for earlier, later in zip(edges, edges[1:])):
                raise ValueError(f"границы измерения {dim} должны возрастать: {edges}")

    @property
    def num_dims(self) -> int:
        return len(self.boundaries)

    @property
    def bins_per_dim(self) -> tuple[int, ...]:
        return tuple(len(edges) + 1 for edges in self.boundaries)

    @property
    def num_states(self) -> int:
        return int(np.prod(self.bins_per_dim))

    @property
    def strides(self) -> tuple[int, ...]:
        """Множители свёртки многомерного номера ячейки в один индекс."""
        strides: list[int] = []
        acc = 1
        for count in reversed(self.bins_per_dim):
            strides.append(acc)
            acc *= count
        return tuple(reversed(strides))

    def cell(self, obs: np.ndarray) -> np.ndarray:
        """``(B, D) -> (B, D)`` номер ячейки по каждому измерению.

        Номер считается как число границ, которые значение превысило.
        Ровно та же формула воспроизводится в графе ONNX
        (:class:`labrl.nets.discretized.DiscretizedQTable`), поэтому изменение
        правила сравнения здесь обязано повторяться и там.
        """
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != self.num_dims:
            raise ValueError(
                f"ожидалось наблюдение формы (B, {self.num_dims}), получено {obs.shape}"
            )

        cells = np.zeros(obs.shape, dtype=np.int64)
        for dim, edges in enumerate(self.boundaries):
            if edges:
                cells[:, dim] = (obs[:, dim, None] > np.asarray(edges)).sum(axis=1)
        return cells

    def index(self, obs: np.ndarray) -> np.ndarray:
        """``(B, D) -> (B,)`` индекс состояния."""
        return (self.cell(obs) * np.asarray(self.strides)).sum(axis=1).astype(np.int64)

    def policy_module(self, q: np.ndarray) -> nn.Module:
        from labrl.nets.discretized import DiscretizedQTable

        return DiscretizedQTable.from_table(self, q)

    def describe(self) -> str:
        """Человекочитаемое описание сетки — для ячейки инспекции в ноутбуке."""
        lines = [f"состояний: {self.num_states} = {' × '.join(str(b) for b in self.bins_per_dim)}"]
        for dim, edges in enumerate(self.boundaries):
            shown = ", ".join(f"{e:+.3f}" for e in edges) if edges else "<нет границ>"
            lines.append(f"  измерение {dim}: {len(edges) + 1} ячеек, границы [{shown}]")
        return "\n".join(lines)
