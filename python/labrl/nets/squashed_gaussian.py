"""Squashed-гауссова политика — политика SAC.

Чем она отличается от политики A2C/PPO
--------------------------------------
У `labrl.nets.gaussian_policy.GaussianPolicyNetwork` действие берётся прямо
из гауссианы и обрезается уже на границе со средой. Для SAC этого мало по двум
причинам.

**Причина 1: границы должны быть внутри распределения.** SAC максимизирует
не только награду, но и энтропию политики, а энтропия неограниченной гауссианы
считается по всей числовой оси — включая действия, которые среда всё равно
обрежет. Метод бы «зарабатывал» энтропию на недостижимых значениях.
Поэтому действие пропускается через ``tanh``::

    u ~ N(μ_θ(s), σ_θ(s)),      a = tanh(u) ∈ (−1, 1)

**Причина 2: замена переменных меняет плотность.** Раз действие — это функция
от `u`, его плотность обязана быть пересчитана по формуле замены переменных::

    log π(a|s) = log N(u | μ, σ) − Σ_i log(1 − tanh²(u_i))

Без второго слагаемого энтропийный член SAC считается по неверной плотности,
и автоподстройка α уводит обучение куда угодно.

**σ зависит от состояния**, в отличие от A2C/PPO: в SAC разведка — часть цели,
и метод должен уметь быть уверенным в одних состояниях и осторожным в других.

Про экспорт в ONNX
------------------
Обёртка экспорта (`labrl.export.onnx_export`) приводит выход политики
к диапазону Unity преобразованием ``clamp(x, −3, 3) / 3``. Политика SAC уже
выдаёт значения в (−1, 1), и деление на 3 сжало бы их втрое. Поэтому
экспортируется не сама политика, а :class:`ExportedSquashedPolicy` — та же
политика, чей выход домножен на 3, чтобы после деления получилось ровно
``tanh(μ)``. Проверяется это тестом, прогоняющим настоящий граф.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from labrl.export.onnx_export import CONTINUOUS_CLIP
from labrl.nets.mlp import build_mlp

#: Границы log σ. Нижняя не даёт распределению выродиться в точку, верхняя —
#: расплыться в шум. Значения из эталонной реализации SAC (Haarnoja et al.).
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

#: Добавка под логарифмом в поправке на tanh. Без неё log(1 − tanh²(u))
#: обращается в −inf при |u| порядка 10, что даёт NaN в функции потерь.
TANH_EPS = 1e-6


class SquashedGaussianPolicy(nn.Module):
    """``obs (B, obs_dim) -> распределение действий в (−1, 1)``.

    Args:
        obs_dim: размерность наблюдения.
        action_dim: размерность непрерывного действия.
        hidden_sizes: скрытые слои общего тела.
        activation: имя активации из :data:`labrl.nets.mlp.ACTIVATIONS`.

    Голов две: среднее и log σ. Обе зависят от состояния — это требование
    метода, а не архитектурный вкус.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        # Тело выдаёт сразу оба параметра: 2·action_dim чисел.
        self.body = build_mlp(obs_dim, 2 * action_dim, hidden_sizes, activation)

    def distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Среднее и log σ, ограниченный безопасным диапазоном."""
        out = self.body(obs)
        mean, log_std = out.chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Действие детерминированной политики: ``tanh(μ)`` ∈ (−1, 1)."""
        mean, _ = self.distribution_params(obs)
        return torch.tanh(mean)

    def sample(self, obs: torch.Tensor, noise: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Действие и его log π(a|s) через reparameterization trick.

        Шум выносится наружу как независимая переменная::

            a = tanh(μ + σ·ε),   ε ~ N(0, I)

        Благодаря этому действие — гладкая функция параметров, и градиент
        Q по действию течёт обратно в веса политики. У REINFORCE такого
        пути нет, и оценка градиента там на порядок шумнее.

        Args:
            obs: ``(B, obs_dim)``.
            noise: ``(B, action_dim)`` — внешний ε. ``None`` — взять из torch.
                Явная передача нужна там, где воспроизводимость прогона
                не должна зависеть от глобального состояния генератора.

        Returns:
            ``(action, log_prob)``: ``(B, action_dim)`` и ``(B,)``.
        """
        mean, log_std = self.distribution_params(obs)
        std = log_std.exp()
        if noise is None:
            noise = torch.randn_like(mean)

        raw = mean + std * noise
        action = torch.tanh(raw)

        # log N(raw | mean, std), просуммированный по компонентам.
        log_prob = (
            -0.5 * ((raw - mean) / std) ** 2
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)

        # Поправка на замену переменных a = tanh(u).
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + TANH_EPS).sum(dim=-1)
        return action, log_prob

    def export_module(self) -> "ExportedSquashedPolicy":
        """Модуль для экспорта в ONNX. См. пояснение в шапке файла."""
        return ExportedSquashedPolicy(self).eval()


class ExportedSquashedPolicy(nn.Module):
    """Политика SAC, подготовленная к приведению диапазона обёрткой экспорта.

    Возвращает ``3·tanh(μ)``: обёртка контракта делит выход на 3 (после
    обрезки в ±3), поэтому в графе окажется ровно ``tanh(μ)`` — то же
    действие, которое выдаёт :meth:`SAC.deterministic_action`.

    Домножение сделано отдельным модулем, а не внутри политики: политика
    обязана оставаться политикой, а знание о контракте ONNX — жить рядом
    с контрактом.
    """

    def __init__(self, policy: SquashedGaussianPolicy) -> None:
        super().__init__()
        self.policy = policy
        self.register_buffer("scale", torch.tensor(float(CONTINUOUS_CLIP)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs) * self.scale
