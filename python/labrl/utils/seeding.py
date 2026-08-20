"""Фиксация источников случайности (требование 12.1).

Сид среды Unity передаётся отдельно — через ``EnvironmentParametersChannel``
(см. :mod:`labrl.envs.side_channels`), потому что генератор Unity живёт
в другом процессе и не управляется отсюда.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Фиксирует ``random``, ``numpy``, ``torch`` (CPU и CUDA).

    Args:
        seed: неотрицательное целое.
        deterministic_torch: включить детерминированные ядра cuDNN. Замедляет
            обучение на GPU, но делает прогон воспроизводимым при одном и том
            же оборудовании.

    Заметка о пределах воспроизводимости: физика Unity (PhysX) недетерминирована
    между запусками даже при одинаковом сиде, поэтому полное совпадение траекторий
    гарантируется только для сред без физики (E01_GridWorld). Это известное
    ограничение, зафиксированное в docs/ASSUMPTIONS.md.
    """
    if seed < 0:
        raise ValueError(f"seed должен быть неотрицательным, получено {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """Локальный генератор numpy — для мест, где глобальное состояние нежелательно."""
    return np.random.default_rng(seed)


def resolve_device(prefer: str = "auto") -> torch.device:
    """``auto`` -> cuda при наличии, иначе cpu. ``cpu``/``cuda`` — как указано."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer not in ("cpu", "cuda"):
        raise ValueError(f"device должен быть auto|cpu|cuda, получено {prefer!r}")
    if prefer == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("запрошен device=cuda, но torch.cuda.is_available() == False")
    return torch.device(prefer)
