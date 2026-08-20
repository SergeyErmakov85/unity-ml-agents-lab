"""Гауссова политика для непрерывных действий.

Почему для непрерывных действий нужна другая конструкция. У value-based
методов политика неявная: ``argmax_a Q(s, a)``. Перебрать все действия можно,
пока их конечное число; в непрерывном пространстве перебора нет. Поэтому
политика задаётся **явно** — сеть выдаёт параметры распределения, из которого
действие берётся сэмплированием::

    a ~ N( μ_θ(s), σ² )

Обучение идёт по градиенту логарифма правдоподобия выбранного действия
(policy gradient), а не по ошибке предсказания ценности.

Про σ. Стандартное отклонение здесь **не зависит от состояния** и хранится
отдельным обучаемым параметром `log_std` — так же, как это делает ML-Agents
и большинство реализаций PPO. Причина практическая: σ(s) даёт политике
возможность «схлопнуть» разброс в удобных состояниях ещё до того, как она
научится в них действовать, и обучение застревает. Отдельный параметр
меняется медленно и ведёт себя как расписание разведки, которое метод
настраивает сам.

Хранится именно **логарифм** σ: он может принимать любые значения, тогда как σ
обязана быть положительной, и оптимизатору не нужно об этом знать.

Про диапазон действий и ONNX. `forward` возвращает **среднее в сырых
единицах**, без клиппинга: ровно этого ждёт обёртка экспорта
(`labrl.export.onnx_export.MLAgentsPolicyWrapper`), которая сама приводит
выход к диапазону Unity преобразованием ``clamp(x, −3, 3) / 3``
(контракт §4.2). То же преобразование обязано применяться и при сборе опыта —
иначе Python и Unity разойдутся; оно вынесено в :func:`to_env_action`.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from labrl.export.onnx_export import CONTINUOUS_CLIP
from labrl.nets.mlp import build_mlp

#: Границы log σ. Нижняя не даёт распределению выродиться в точку (градиент
#: логарифма правдоподобия при σ → 0 уходит в бесконечность), верхняя — стать
#: настолько широким, что политика превращается в случайный шум.
LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0


def to_env_action(raw_action: torch.Tensor | "object") -> "object":
    """Приводит сырое действие политики к диапазону, который принимает Unity.

    Преобразование `clamp(x, −3, 3) / 3` взято не из соображений удобства:
    ровно его выполняет граф ONNX по контракту ML-Agents
    (`docs/04_ONNX_CONTRACT.md`, §4.2). Применять его при сборе опыта
    **обязательно** — иначе обученная политика в Unity будет действовать
    иначе, чем при обучении, и проверка 10.6 провалится без внятной причины.

    Работает и с ``torch.Tensor``, и с ``numpy.ndarray``.
    """
    if isinstance(raw_action, torch.Tensor):
        return torch.clamp(raw_action, -CONTINUOUS_CLIP, CONTINUOUS_CLIP) / CONTINUOUS_CLIP

    import numpy as np

    return np.clip(raw_action, -CONTINUOUS_CLIP, CONTINUOUS_CLIP) / CONTINUOUS_CLIP


class GaussianPolicyNetwork(nn.Module):
    """Сеть политики: ``obs (B, obs_dim) -> среднее (B, action_dim)``.

    Соответствует протоколу :class:`labrl.nets.protocols.PolicyNetwork`
    и одновременно пригодна к экспорту: `forward` возвращает только среднее.

    Args:
        obs_dim: размерность наблюдения.
        action_dim: размерность непрерывного действия.
        hidden_sizes: скрытые слои.
        activation: имя активации из :data:`labrl.nets.mlp.ACTIVATIONS`.
        log_std_init: начальный log σ. −0.5 даёт σ ≈ 0.6: разведка заметная,
            но не заглушающая сигнал.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "tanh",
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.body = build_mlp(obs_dim, action_dim, hidden_sizes, activation)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Среднее политики в сырых единицах, до приведения к диапазону Unity."""
        return self.body(obs)

    def clamped_log_std(self) -> torch.Tensor:
        """log σ, ограниченный безопасным диапазоном."""
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Распределение действий N(μ(s), σ) с независимыми компонентами."""
        mean = self(obs)
        std = self.clamped_log_std().exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """log π(a|s), просуммированный по компонентам действия. ``(B,)``.

        Суммирование, а не усреднение: компоненты независимы, и логарифм
        совместной плотности — сумма логарифмов. Усреднение изменило бы
        масштаб градиента в `action_dim` раз.
        """
        return self.distribution(obs).log_prob(action).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Энтропия политики, нат. Не зависит от состояния, как и σ.

        Для N(μ, σ): H = Σ_i ( ln σ_i + ½·ln(2πe) ).
        """
        return (self.clamped_log_std() + 0.5 * math.log(2.0 * math.pi * math.e)).sum()
