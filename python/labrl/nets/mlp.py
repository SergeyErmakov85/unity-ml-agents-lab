"""Простейшие полносвязные сети — значения по умолчанию, а не «правильный ответ».

Пользователь ноутбука волен заменить любую из них своей архитектурой: алгоритмы
принимают любой ``nn.Module``, удовлетворяющий протоколам из
:mod:`labrl.nets.protocols` (требование 8.4).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

#: Поддерживаемые активации. Список сознательно короткий: учебный код.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: Sequence[int] = (128, 128),
    activation: str = "relu",
) -> nn.Sequential:
    """Собирает ``Linear -> act -> ... -> Linear`` без активации на выходе."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation должна быть одной из {sorted(ACTIVATIONS)}, получено {activation!r}")
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MLPQNetwork(nn.Module):
    """Q-сеть: ``obs (B, obs_dim) -> Q (B, num_actions)``.

    Соответствует протоколу :class:`labrl.nets.protocols.QNetwork`.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.body = build_mlp(obs_dim, num_actions, hidden_sizes, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)


class MLPPolicyNetwork(nn.Module):
    """Сеть политики: ``obs (B, obs_dim) -> логиты (B, action_dim)``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.body = build_mlp(obs_dim, action_dim, hidden_sizes, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)


class MLPContinuousQNetwork(nn.Module):
    """Критик непрерывного управления: ``(obs, action) -> Q`` формы ``(B,)``.

    Отличие от :class:`MLPQNetwork` принципиальное. При дискретных действиях
    сеть выдаёт Q сразу для всех действий, и `argmax` берётся перебором.
    В непрерывном пространстве перебора нет, поэтому действие подаётся
    **на вход** вместе с наблюдением, а выход — одно число.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.body = build_mlp(obs_dim + action_dim, 1, hidden_sizes, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([obs, action], dim=-1)).squeeze(-1)


class MLPValueNetwork(nn.Module):
    """Критик: ``obs (B, obs_dim) -> V (B, 1)``."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.body = build_mlp(obs_dim, 1, hidden_sizes, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)
