"""Реестр сред лаборатории.

Единственное место, где перечислены присвоенные коды `E##`. Нужен, чтобы
опечатка в идентификаторе среды находилась до запуска Unity, а не после
минуты ожидания подключения.

Реестр **не** дублирует ENV_SPEC.md: здесь только то, что нужно Python-стороне
для подключения и проверки. Размерности наблюдений и действий берутся из живой
среды (`BehaviorSpec`), потому что источник истины для них — Unity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from labrl.utils.config import repo_root


@dataclass(frozen=True)
class EnvEntry:
    """Запись реестра."""

    env_id: str
    title: str
    category: str
    status: str  # PLANNED | IN_PROGRESS | DONE

    @property
    def unity_folder(self) -> Path:
        return repo_root() / "unity" / "MLAgentsLab" / "Assets" / "Envs" / self.env_id

    @property
    def build_path(self) -> Path:
        """Путь к исполняемому файлу headless-сборки по соглашению каталогов."""
        return repo_root() / "builds" / self.env_id / f"{self.env_id}.exe"

    @property
    def env_spec_path(self) -> Path:
        return self.unity_folder / "ENV_SPEC.md"


#: Коды присвоены и неизменяемы (правило 5.3, решение по вопросу Q2).
#: Состав — по фактическим урокам курса, см. docs/02_LESSON_MAP.md.
REGISTRY: dict[str, EnvEntry] = {
    entry.env_id: entry
    for entry in [
        EnvEntry("E00_Bandit", "Многорукий бандит", "Основы", "PLANNED"),
        EnvEntry("E01_GridWorld", "GridWorld 5×5", "Табличные методы", "IN_PROGRESS"),
        EnvEntry("E02_CartPoleUnity", "Балансировка шеста", "Табличные методы", "PLANNED"),
        EnvEntry("E03_RollerBall", "Докатись до цели", "Value-based", "IN_PROGRESS"),
        EnvEntry("E04_BallBalance", "Удержание шара на платформе", "Непрерывное управление", "PLANNED"),
        EnvEntry("E05_FoodCollector", "Сбор еды", "Policy gradient", "PLANNED"),
        EnvEntry("E06_Hunter3D", "Преследование в 3D", "Actor-Critic", "PLANNED"),
        EnvEntry("E07_RacingCar", "Гоночная трасса", "Непрерывное управление", "PLANNED"),
        EnvEntry("E08_SoccerArena", "Футбольная арена", "Мультиагентность", "PLANNED"),
        EnvEntry("E09_CurriculumMaze", "Лабиринт с curriculum", "Curriculum / DR", "PLANNED"),
        EnvEntry("E10_Imitation", "Имитационное обучение", "Имитация", "PLANNED"),
        EnvEntry("E11_Research", "Исследовательский слот", "Исследования", "PLANNED"),
    ]
}


def get(env_id: str) -> EnvEntry:
    """Возвращает запись реестра или падает с внятным сообщением."""
    try:
        return REGISTRY[env_id]
    except KeyError:
        raise KeyError(
            f"среда {env_id!r} не зарегистрирована; известны: {sorted(REGISTRY)}.\n"
            "Новая среда добавляется в этот реестр и в docs/02_LESSON_MAP.md."
        ) from None


def default_build_path(env_id: str) -> Path:
    """Путь к билду по соглашению. Существование файла не проверяется."""
    return get(env_id).build_path
