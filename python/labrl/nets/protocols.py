"""Протоколы авторских сетей (требование 8.4).

Смысл модуля: алгоритм не знает архитектуру. Пользователь пишет **свой**
``nn.Module`` прямо в ноутбуке (ячейка 5 по разделу 9) и передаёт его
в конструктор алгоритма. Единственное, что алгоритм требует, — соблюдение
формы входов и выходов, описанной здесь.

Соглашения по формам (действуют во всём пакете):

* ``B`` — размер батча;
* ``obs`` — ``FloatTensor`` формы ``(B, obs_dim)`` для векторных наблюдений
  либо ``(B, C, H, W)`` для визуальных (NCHW — как требует контракт ONNX, §1);
* дискретные действия — ``LongTensor`` формы ``(B, num_branches)`` с индексами;
* непрерывные действия — ``FloatTensor`` формы ``(B, continuous_size)``
  в диапазоне ``[-1, 1]`` (контракт ONNX, §4.2).

Протоколы объявлены ``runtime_checkable`` — этого достаточно для проверки
наличия методов; формы проверяются тестами и ``assert``-ами в ноутбуках.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class QNetwork(Protocol):
    """Сеть ценности действий для value-based методов (DQN и производные).

    Работает с **одной** дискретной веткой действий: это покрывает все среды
    курса и позволяет не усложнять учебный код мультиветвевой логикой.
    """

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Q-значения всех действий.

        Args:
            obs: ``(B, obs_dim)``.

        Returns:
            ``(B, num_actions)`` — Q(s, a) для каждого действия.
        """
        ...


@runtime_checkable
class PolicyNetwork(Protocol):
    """Сеть политики для policy-gradient методов.

    Возвращает **логиты** (дискретный случай) или параметры распределения
    (непрерывный случай). Сэмплирование и детерминированный выбор делает
    алгоритм, а не сеть, — иначе экспорт в ONNX пришлось бы дублировать.
    """

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Args: ``obs`` ``(B, obs_dim)``. Returns: ``(B, action_dim)``."""
        ...


@runtime_checkable
class ValueNetwork(Protocol):
    """Сеть ценности состояния V(s) — критик."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Args: ``obs`` ``(B, obs_dim)``. Returns: ``(B, 1)``."""
        ...


def check_q_network(net: torch.nn.Module, obs_dim: int, num_actions: int, device: torch.device | str = "cpu") -> None:
    """Проверяет форму выхода Q-сети прогоном батча из двух наблюдений.

    Вызывается в ноутбуке сразу после определения архитектуры: ошибка формы,
    найденная здесь, стоит секунду, а найденная в цикле обучения — час.
    """
    net = net.to(device)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(2, obs_dim, device=device))
    if was_training:
        net.train()
    if out.shape != (2, num_actions):
        raise ValueError(
            f"QNetwork должна возвращать (B, num_actions) = (2, {num_actions}), получено {tuple(out.shape)}"
        )
