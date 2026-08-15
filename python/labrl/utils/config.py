"""Загрузка и валидация конфигов экспериментов (`configs/E##_<Name>__<algo>.yaml`).

Конфиг — единственный источник значений, влияющих на результат (требование 8.6).
Структура блоков описана в Приложении A инструкции проекта.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Блоки верхнего уровня, обязательные в любом конфиге эксперимента.
REQUIRED_SECTIONS = ("experiment", "env", "network", "algo", "eval", "success_criteria", "export")


def repo_root() -> Path:
    """Корень репозитория. Определяется по расположению пакета, а не по CWD.

    labrl лежит в ``<repo>/python/labrl``, поэтому корень — на два уровня выше.
    Требование 9.5 запрещает абсолютные пути вне корня репозитория, и эта
    функция — единственное место, где корень вычисляется.
    """
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Приводит путь из конфига к абсолютному относительно корня репозитория."""
    p = Path(path)
    return p if p.is_absolute() else (repo_root() / p)


@dataclass
class SuccessCriteria:
    """Числовой критерий приёмки примера (требование 12.4)."""

    metric: str
    threshold: float
    within_steps: int


@dataclass
class ExperimentConfig:
    """Полный конфиг прогона. Хранит и разобранные поля, и исходный словарь.

    Исходный словарь (``raw``) копируется в каталог прогона без изменений —
    так конфиг остаётся точным описанием того, что было запущено (требование 11.1).
    """

    raw: dict[str, Any]
    path: Path | None = None

    experiment: dict[str, Any] = field(init=False)
    env: dict[str, Any] = field(init=False)
    network: dict[str, Any] = field(init=False)
    algo: dict[str, Any] = field(init=False)
    eval: dict[str, Any] = field(init=False)
    export: dict[str, Any] = field(init=False)
    success_criteria: SuccessCriteria = field(init=False)

    def __post_init__(self) -> None:
        missing = [s for s in REQUIRED_SECTIONS if s not in self.raw]
        if missing:
            raise ValueError(f"в конфиге отсутствуют обязательные блоки: {missing}")
        for section in ("experiment", "env", "network", "algo", "eval", "export"):
            setattr(self, section, copy.deepcopy(self.raw[section]))
        self.success_criteria = SuccessCriteria(**self.raw["success_criteria"])

        for key in ("env_id", "algo", "seeds", "total_steps"):
            if key not in self.experiment:
                raise ValueError(f"experiment.{key} обязателен")

        env_id = self.experiment["env_id"]
        if not _is_valid_env_id(env_id):
            raise ValueError(
                f"experiment.env_id={env_id!r} не соответствует схеме E##_<PascalName> (правило 5.3)"
            )

    @property
    def env_id(self) -> str:
        """Идентификатор среды. Он же — Behavior Name в Unity (правило 5.3)."""
        return self.experiment["env_id"]

    @property
    def algo_name(self) -> str:
        return self.experiment["algo"]

    @property
    def seeds(self) -> list[int]:
        return list(self.experiment["seeds"])

    @property
    def total_steps(self) -> int:
        return int(self.experiment["total_steps"])


def _is_valid_env_id(env_id: object) -> bool:
    if not isinstance(env_id, str) or len(env_id) < 5:
        return False
    if env_id[0] != "E" or not env_id[1:3].isdigit() or env_id[3] != "_":
        return False
    name = env_id[4:]
    return bool(name) and name[0].isupper() and name.isalnum()


def load_config(path: str | Path) -> ExperimentConfig:
    """Читает YAML-конфиг эксперимента и проверяет обязательные блоки."""
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"конфиг {p} должен быть YAML-словарём, получено {type(raw).__name__}")
    return ExperimentConfig(raw=raw, path=p)


def dump_config(cfg: ExperimentConfig, dest: str | Path) -> Path:
    """Сохраняет исходный конфиг рядом с результатами прогона."""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg.raw, fh, allow_unicode=True, sort_keys=False)
    return out
