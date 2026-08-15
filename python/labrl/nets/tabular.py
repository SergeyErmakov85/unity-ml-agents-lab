"""Табличная политика как экспортируемый в ONNX модуль.

Задача. Табличный метод хранит `Q` матрицей `(num_states, num_actions)`,
а Unity умеет исполнять только граф ONNX. Мост между ними — наблюдение:
среда отдаёт состояние **one-hot вектором**, а умножение one-hot на матрицу
выбирает строку::

    onehot(s) @ Qᵀ  ==  Q[s]

Значит линейный слой без смещения с весами `Q` вычисляет ровно ту же строку
таблицы. В Unity уезжает **та же самая** политика, а не её приближение —
никакой аппроксимации и никакой потери точности, кроме перевода float64 → float32.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class OneHotQTable(nn.Module):
    """Линейный слой, эквивалентный таблице Q при one-hot наблюдении.

    Соответствует протоколу :class:`labrl.nets.protocols.QNetwork`:
    ``forward(obs: (B, num_states)) -> (B, num_actions)``.
    """

    def __init__(self, num_states: int, num_actions: int) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.table = nn.Linear(self.num_states, self.num_actions, bias=False)

    @classmethod
    def from_table(cls, q: np.ndarray) -> "OneHotQTable":
        """Собирает модуль из таблицы ``(num_states, num_actions)``."""
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 2:
            raise ValueError(f"таблица Q должна быть двумерной, получено {q.shape}")
        module = cls(q.shape[0], q.shape[1])
        with torch.no_grad():
            # nn.Linear хранит веса как (out_features, in_features), то есть Qᵀ.
            module.table.weight.copy_(torch.from_numpy(q.T))
        return module.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.table(obs)

    def to_table(self) -> np.ndarray:
        """Обратное преобразование — для сверки экспортированного модуля с таблицей."""
        return self.table.weight.detach().cpu().numpy().T.copy()
