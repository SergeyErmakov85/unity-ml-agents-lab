"""Сохранение и загрузка чекпойнтов обучения."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Записывает чекпойнт атомарно: сначала во временный файл, затем replace.

    Атомарность важна: обрыв обучения на середине записи не должен оставлять
    повреждённый файл вместо предыдущего рабочего чекпойнта.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)
    return p


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Читает чекпойнт. ``map_location='cpu'`` — чтобы файл с GPU читался везде."""
    return torch.load(Path(path), map_location=map_location)
