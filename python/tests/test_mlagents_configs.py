"""Эталонные конфиги штатного тренера обязаны быть валидны для него самого.

Зачем этот тест. Конфиги в `configs/mlagents/` существуют ради пункта 12.5
инструкции: при подозрении на ошибку должно быть с чем сравнить. Толку
от них ноль, если `mlagents-learn` отвергнет их при запуске — а узнать
об этом хочется не в тот момент, когда что-то уже сломалось.

Схема берётся из **установленного пакета**, а не переписывается здесь
(правило 16.4): единственный источник истины о том, какие ключи допустимы, —
`mlagents.trainers.settings`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "mlagents"


def config_paths() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.fixture(scope="module")
def run_options():
    """Класс настроек штатного тренера с зарегистрированными типами тренеров.

    Две тонкости, обе найдены запуском.

    **Первая.** Без `register_trainer_plugins()` настройки не знают ни одного
    `trainer_type` и отвергают даже правильный конфиг с сообщением
    «Invalid trainer type ppo was found» — ошибка, которую легко принять
    за ошибку конфига.

    **Вторая.** Импорт `mlagents.torch_utils` выполняет
    ``torch.set_default_device(...)`` и при наличии CUDA переключает
    устройство по умолчанию **глобально, на весь процесс**
    (`mlagents/torch_utils/torch.py`, строка 55). Соседние тесты, создающие
    тензоры на CPU и умножающие их на параметры сети, после этого падают
    с «Expected all tensors to be on the same device». Поэтому фикстура
    возвращает устройство по умолчанию обратно.
    """
    import torch

    from mlagents.plugins.trainer_type import register_trainer_plugins
    from mlagents.trainers.settings import RunOptions

    register_trainer_plugins()
    try:
        yield RunOptions
    finally:
        # None возвращает поведение по умолчанию (torch >= 2.0).
        torch.set_default_device(None)


def test_reference_configs_exist():
    paths = config_paths()
    assert paths, f"в {CONFIG_DIR} нет ни одного эталонного конфига"


@pytest.mark.parametrize("path", config_paths(), ids=lambda p: p.stem)
def test_reference_config_is_accepted_by_mlagents(path, run_options):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    run_options.from_dict(raw)


@pytest.mark.parametrize("path", config_paths(), ids=lambda p: p.stem)
def test_behavior_name_matches_the_environment_id(path):
    """Ключ `behaviors` обязан совпадать с именем файла — то есть
    с идентификатором среды (правило 5.3).

    Расхождение здесь означает, что штатный тренер подключится к сцене
    и не увидит агентов — без внятной ошибки.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = list(raw["behaviors"])
    assert names == [path.stem], f"{path.name}: behaviors {names}, ожидалось [{path.stem}]"


@pytest.mark.parametrize("path", config_paths(), ids=lambda p: p.stem)
def test_behavior_name_has_no_team_suffix(path):
    """Суффикс `?team=` в конфиге писать не нужно: штатный тренер добавляет
    его сам. Написанный руками, он даёт поведение, которого в сцене нет."""
    assert "?team=" not in path.read_text(encoding="utf-8").split("behaviors:")[1][:200]
