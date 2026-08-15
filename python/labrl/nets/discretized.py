"""Табличная политика с дискретизацией **внутри** графа ONNX.

Задача. В `E02_CartPoleUnity` наблюдение непрерывно, а метод — табличный:
между ними стоит сетка дискретизации. Если оставить сетку в Python, Unity
получит модель, которая читает сырое наблюдение и истолкует его иначе, чем
обучение. Требование 10.7 прямо это запрещает: любое преобразование входа
обязано быть частью графа.

Здесь сетка выражена операциями, которые понимает opset 9 — тем самым,
с которым ML-Agents экспортирует модели (`docs/04_ONNX_CONTRACT.md`):

1. **номер ячейки по измерению** — число превышенных границ::

       cell_d = Σ_j  [ obs_d > boundary_dj ]        (Greater + Cast + ReduceSum)

2. **свёртка в один индекс** — скалярное произведение со шагами::

       state = Σ_d  cell_d · stride_d               (Mul + ReduceSum)

3. **выбор строки таблицы** — one-hot без операции one-hot::

       onehot_s = ReLU(1 − |state − s|)             (Sub + Abs + Sub + Relu)
       Q        = onehot @ table                     (MatMul)

Третий шаг — единственный неочевидный. Прямого `OneHot` в opset 9 для этой
схемы нет, а `Equal` на вещественных типах до opset 11 не определён. Но
`state` по построению целое, поэтому «треугольная» функция ``ReLU(1 − |Δ|)``
даёт ровно единицу при совпадении и ровно ноль при любом другом целом
расстоянии. Никакого приближения здесь нет.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from labrl.envs.state_encoders import BoxDiscretizer

#: Значение-заполнитель для выравнивания границ по измерениям.
#: Наблюдение никогда его не превысит, поэтому «лишние» границы не считаются.
PAD_BOUNDARY = 1e30


class DiscretizedQTable(nn.Module):
    """Q-таблица, принимающая **сырое** непрерывное наблюдение.

    Соответствует протоколу :class:`labrl.nets.protocols.QNetwork`:
    ``forward(obs: (B, num_dims)) -> (B, num_actions)``.

    Args:
        discretizer: сетка, по которой обучалась таблица.
        num_actions: число действий (одна дискретная ветка).
    """

    def __init__(self, discretizer: BoxDiscretizer, num_actions: int) -> None:
        super().__init__()
        self.num_dims = discretizer.num_dims
        self.num_states = discretizer.num_states
        self.num_actions = int(num_actions)

        widths = [len(edges) for edges in discretizer.boundaries]
        max_width = max(widths) if widths else 0

        # Границы выравниваются в матрицу (D, max_width): разные измерения имеют
        # разное число границ, а граф работает с прямоугольными тензорами.
        padded = np.full((self.num_dims, max(max_width, 1)), PAD_BOUNDARY, dtype=np.float32)
        for dim, edges in enumerate(discretizer.boundaries):
            if edges:
                padded[dim, : len(edges)] = np.asarray(edges, dtype=np.float32)

        self.register_buffer("boundaries", torch.from_numpy(padded))
        self.register_buffer(
            "strides", torch.tensor(discretizer.strides, dtype=torch.float32)
        )
        self.register_buffer(
            "state_ids", torch.arange(self.num_states, dtype=torch.float32)
        )
        self.register_buffer(
            "table", torch.zeros(self.num_states, self.num_actions, dtype=torch.float32)
        )

    @classmethod
    def from_table(cls, discretizer: BoxDiscretizer, q: np.ndarray) -> "DiscretizedQTable":
        """Собирает модуль из сетки и таблицы ``(num_states, num_actions)``."""
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 2:
            raise ValueError(f"таблица Q должна быть двумерной, получено {q.shape}")
        if q.shape[0] != discretizer.num_states:
            raise ValueError(
                f"в таблице {q.shape[0]} состояний, сетка задаёт {discretizer.num_states}"
            )

        module = cls(discretizer, q.shape[1])
        with torch.no_grad():
            module.table.copy_(torch.from_numpy(q))
        return module.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 1. Номер ячейки по каждому измерению — число превышенных границ.
        cells = (obs.unsqueeze(2) > self.boundaries.unsqueeze(0)).to(obs.dtype).sum(dim=2)

        # 2. Свёртка многомерного номера в один индекс состояния.
        state = (cells * self.strides.unsqueeze(0)).sum(dim=1, keepdim=True)

        # 3. One-hot без операции one-hot: индекс целый, поэтому «треугольник»
        #    ReLU(1 − |Δ|) равен единице ровно в своей позиции.
        onehot = torch.relu(1.0 - torch.abs(state - self.state_ids.unsqueeze(0)))

        return onehot.matmul(self.table)

    def to_table(self) -> np.ndarray:
        """Обратное преобразование — для сверки экспортированного модуля с таблицей."""
        return self.table.detach().cpu().numpy().copy()
