"""Гибридная политика: сетка + вектор на входе, две группы действий на выходе.

Зачем отдельный модуль. До `E05_FoodCollector` все среды лаборатории отдавали
наблюдение одним вектором, и политика была обычным MLP. Здесь два отличия
сразу:

**Вход.** `GridSensor` отдаёт тензор ``(C, H, W)`` — карту вокруг агента,
по каналу на сорт объекта. Приклеить её к вектору проприоцепции нельзя:
у карты есть пространственная структура, и разрушать её вытягиванием в строку
значит выбрасывать то, ради чего сенсор и заведён. Поэтому карта идёт
в свёрточную часть, вектор — мимо неё, и объединяются они уже признаками.

**Выход.** Пространство действий гибридное: движение непрерывно, ускорение —
переключатель. Политика возвращает **пару** ``(среднее, логиты)``, и ровно
такую пару ждёт обёртка экспорта ONNX для гибридного контракта.

Про формат ``(C, H, W)``. Это не наш выбор и не преобразование: `ObservationSpec.Visual`
в ML-Agents объявляет форму именно как ``(channels, height, width)``
(проверено в `com.unity.ml-agents@4.0.3`, `Runtime/Sensors/ObservationSpec.cs`),
а `torch.nn.Conv2d` ждёт ``(B, C, H, W)``. Никакой перестановки осей
не требуется — и это стоит проверить при первом же экспорте, потому что
ошибка здесь не даёт ни исключения, ни предупреждения.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from labrl.nets.mlp import ACTIVATIONS, build_mlp

#: Границы log σ непрерывной части — те же соображения, что в
#: :mod:`labrl.nets.gaussian_policy`.
LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0


class GridHybridPolicy(nn.Module):
    """``(grid, vector) -> (среднее непрерывных действий, логиты дискретных)``.

    Args:
        grid_shape: форма сеточного наблюдения ``(C, H, W)``.
        vector_dim: размерность векторного наблюдения.
        continuous_dim: размерность непрерывной части действия.
        discrete_branches: размеры дискретных веток.
        conv_channels: число каналов свёрточных слоёв.
        hidden_sizes: скрытые слои общего тела после объединения признаков.
        activation: имя активации из :data:`labrl.nets.mlp.ACTIVATIONS`.
        log_std_init: начальный log σ непрерывной части.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        vector_dim: int,
        continuous_dim: int,
        discrete_branches: Sequence[int],
        conv_channels: Sequence[int] = (16, 32),
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation должна быть одной из {sorted(ACTIVATIONS)}")

        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.vector_dim = int(vector_dim)
        self.continuous_dim = int(continuous_dim)
        self.discrete_branches = tuple(int(b) for b in discrete_branches)

        act = ACTIVATIONS[activation]
        channels, height, width = self.grid_shape

        layers: list[nn.Module] = []
        previous = channels
        for out_channels in conv_channels:
            # padding=1 при ядре 3 сохраняет размер карты: сетка и без того
            # маленькая (8×8), и ужимать её нечем.
            layers += [nn.Conv2d(previous, out_channels, kernel_size=3, padding=1), act()]
            previous = out_channels
        layers.append(nn.Flatten())
        self.grid_encoder = nn.Sequential(*layers)

        grid_features = previous * height * width
        self.trunk = build_mlp(grid_features + self.vector_dim,
                               hidden_sizes[-1], hidden_sizes[:-1], activation)
        self.trunk_activation = act()

        self.continuous_mean = nn.Linear(hidden_sizes[-1], self.continuous_dim)
        self.log_std = nn.Parameter(torch.full((self.continuous_dim,), float(log_std_init)))
        self.discrete_logits = nn.Linear(hidden_sizes[-1], sum(self.discrete_branches))

    # --- признаки --------------------------------------------------------

    def features(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Общие признаки: свёртка по карте плюс вектор проприоцепции."""
        encoded = self.grid_encoder(grid)
        return self.trunk_activation(self.trunk(torch.cat([encoded, vector], dim=1)))

    def forward(self, grid: torch.Tensor, vector: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Пара ``(среднее, логиты)`` — формат, который ждёт обёртка экспорта."""
        hidden = self.features(grid, vector)
        return self.continuous_mean(hidden), self.discrete_logits(hidden)

    def clamped_log_std(self) -> torch.Tensor:
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    # --- распределения ---------------------------------------------------

    def split_logits(self, logits: torch.Tensor) -> list[torch.Tensor]:
        """Логиты по веткам. У каждой ветки — своё независимое распределение."""
        return list(torch.split(logits, list(self.discrete_branches), dim=1))

    def log_prob(
        self,
        grid: torch.Tensor,
        vector: torch.Tensor,
        continuous_action: torch.Tensor,
        discrete_action: torch.Tensor,
    ) -> torch.Tensor:
        """log π(a|s) гибридного действия. ``(B,)``.

        Действие состоит из независимых частей, поэтому логарифмы
        складываются: непрерывная часть — гауссиана, каждая дискретная
        ветка — категориальное распределение.
        """
        mean, logits = self(grid, vector)
        std = self.clamped_log_std().exp().expand_as(mean)
        total = torch.distributions.Normal(mean, std).log_prob(continuous_action).sum(dim=-1)

        for branch, branch_logits in enumerate(self.split_logits(logits)):
            distribution = torch.distributions.Categorical(logits=branch_logits)
            total = total + distribution.log_prob(discrete_action[:, branch])
        return total

    def entropy(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Энтропия политики, нат: сумма по непрерывной и дискретным частям."""
        _, logits = self(grid, vector)
        continuous = (self.clamped_log_std() + 0.5 * math.log(2.0 * math.pi * math.e)).sum()
        discrete = sum(
            torch.distributions.Categorical(logits=branch_logits).entropy().mean()
            for branch_logits in self.split_logits(logits)
        )
        return continuous + discrete


class GridValueNetwork(nn.Module):
    """Baseline ``V(s)`` для сеточного наблюдения: ``(grid, vector) -> (B, 1)``.

    Своя свёрточная часть, а не общая с политикой. Общее тело экономит
    параметры, но требует подбирать вес между двумя градиентами, которые
    тянут его в разные стороны, — а выигрыш на учебной задаче
    пренебрежимо мал.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        vector_dim: int,
        conv_channels: Sequence[int] = (16, 32),
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = ACTIVATIONS[activation]
        channels, height, width = tuple(int(x) for x in grid_shape)

        layers: list[nn.Module] = []
        previous = channels
        for out_channels in conv_channels:
            layers += [nn.Conv2d(previous, out_channels, kernel_size=3, padding=1), act()]
            previous = out_channels
        layers.append(nn.Flatten())
        self.grid_encoder = nn.Sequential(*layers)
        self.head = build_mlp(previous * height * width + int(vector_dim), 1,
                              hidden_sizes, activation)

    def forward(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.grid_encoder(grid), vector], dim=1))
