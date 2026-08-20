"""Сквозная согласованность репозитория.

Проверяет то, что легко разъезжается между Unity, Python, конфигами
и документами и что ни один отдельный тест не ловит: у каждой среды есть
папка, сцена, Setup-скрипт, строка-контракт, конфиг, ноутбук и карточка,
а каждый конфиг ссылается на существующую среду по соглашениям
`docs/03_CONVENTIONS.md`.

Такие проверки обычно живут в CI как отдельный скрипт. Здесь они — тесты,
потому что запускаются тем же `pytest`, что и всё остальное, и потому
не забываются.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
import yaml

from labrl.envs.registry import REGISTRY
from labrl.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
ENVS = ROOT / "unity" / "MLAgentsLab" / "Assets" / "Envs"
CONFIGS = ROOT / "configs"

#: Несколько стратегий бандита описаны одной карточкой.
ALGO_CARD_ALIASES = {"eps_greedy": "bandits", "ucb": "bandits", "thompson": "bandits"}

ENV_IDS = sorted(REGISTRY)


def our_configs() -> list[Path]:
    return sorted(CONFIGS.glob("*.yaml"))


# --- состав среды -------------------------------------------------------


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_environment_folder_is_complete(env_id):
    folder = ENVS / env_id
    assert folder.is_dir(), f"нет папки {folder}"
    assert (folder / "ENV_SPEC.md").is_file(), "нет ENV_SPEC.md"
    assert any((folder / "Editor").glob("*.cs")), "нет Setup-скрипта в Editor/"
    assert any((folder / "Scripts").glob("*.cs")), "нет скриптов в Scripts/"
    assert (folder / "Models").is_dir(), "нет папки Models/"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_environment_has_exactly_one_scene(env_id):
    """Ровно одна сцена: `BuildScript` и `SceneValidator` считают вторую
    ошибкой, потому что «собрать какую-нибудь» означает получить билд,
    в котором Python не найдёт нужное поведение."""
    scenes = list((ENVS / env_id / "Scenes").glob("*.unity"))
    assert len(scenes) == 1, f"сцен {len(scenes)}: {[s.name for s in scenes]}"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_spec_has_validator_contract(env_id):
    """Без строки-контракта `SceneValidator` не сверит размерности
    со спецификацией (требование 7.7)."""
    text = (ENVS / env_id / "ENV_SPEC.md").read_text(encoding="utf-8")
    assert re.search(r"<!--\s*validator:", text), "нет строки <!-- validator: … -->"


def test_registry_matches_disk():
    """Реестр Python и папки Unity обязаны совпадать в обе стороны.

    Среда, которой нет в реестре, не откроется из Python по имени;
    запись реестра без папки даст `FileNotFoundError` на билде.
    """
    on_disk = {p.name for p in ENVS.iterdir() if p.is_dir()}
    assert on_disk == set(REGISTRY), (
        f"только на диске: {sorted(on_disk - set(REGISTRY))}, "
        f"только в реестре: {sorted(set(REGISTRY) - on_disk)}"
    )


# --- конфиги ------------------------------------------------------------


@pytest.mark.parametrize("path", our_configs(), ids=lambda p: p.stem)
def test_config_loads_and_points_at_a_known_environment(path):
    cfg = load_config(path)
    assert cfg.env_id in REGISTRY, f"env_id {cfg.env_id} нет в реестре"


@pytest.mark.parametrize("path", our_configs(), ids=lambda p: p.stem)
def test_config_file_name_follows_the_convention(path):
    """`E##_<Name>__<algo>.yaml` — по `docs/03_CONVENTIONS.md`, §6."""
    cfg = load_config(path)
    assert path.stem.startswith(cfg.env_id + "__"), (
        f"имя файла не начинается с {cfg.env_id}__"
    )


@pytest.mark.parametrize("path", our_configs(), ids=lambda p: p.stem)
def test_config_paths_follow_the_convention(path):
    """Путь билда и путь экспорта — по соглашению каталогов (§1).

    Ошибка здесь означает, что модель уедет в чужую папку либо билд
    будет искаться там, где его нет, — и то и другое обнаружится
    только при запуске.
    """
    cfg = load_config(path)
    build = str(cfg.env.get("build_path", "")).replace("\\", "/")
    assert build == f"builds/{cfg.env_id}/{cfg.env_id}.exe", f"build_path: {build}"

    onnx = str(cfg.export.get("onnx_path", "")).replace("\\", "/")
    assert f"Envs/{cfg.env_id}/Models/" in onnx, f"onnx_path: {onnx}"


@pytest.mark.parametrize("path", our_configs(), ids=lambda p: p.stem)
def test_config_has_success_criteria(path):
    """Блок `success_criteria` обязателен: он же критерий приёмки (12.4)."""
    cfg = load_config(path)
    assert cfg.success_criteria.metric
    assert cfg.success_criteria.threshold is not None


# --- документация -------------------------------------------------------


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_environment_has_a_card(env_id):
    assert (ROOT / "docs" / "envs" / f"{env_id}.md").is_file()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_environment_has_at_least_one_notebook(env_id):
    names = {p.name for p in (ROOT / "notebooks").glob("*.ipynb")}
    assert any(n.startswith(env_id + "__") for n in names), (
        f"нет ни одного {env_id}__*.ipynb"
    )


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_environment_has_a_reference_trainer_config(env_id):
    """Эталонный конфиг штатного тренера — требование 12.5 (`SHOULD`)."""
    path = CONFIGS / "mlagents" / f"{env_id}.yaml"
    assert path.is_file(), f"нет configs/mlagents/{env_id}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(raw["behaviors"]) == [env_id]


def test_every_algorithm_has_a_card():
    missing = []
    for path in our_configs():
        algo = load_config(path).algo_name
        name = ALGO_CARD_ALIASES.get(algo, algo)
        if not (ROOT / "docs" / "algos" / f"{name}.md").is_file():
            missing.append(f"{algo} → docs/algos/{name}.md")
    assert not missing, f"нет карточек: {missing}"


# --- гигиена текста -----------------------------------------------------


def text_files() -> list[Path]:
    paths = list(ROOT.glob("*.md"))
    paths += list((ROOT / "docs").rglob("*.md"))
    paths += list(CONFIGS.rglob("*.yaml"))
    return sorted(paths)


@pytest.mark.parametrize("path", text_files(), ids=lambda p: p.name)
def test_no_control_characters_in_text(path):
    """Управляющий символ в тексте почти всегда означает съеденный обратный
    слэш: `scripts\\train.py` превращается в `scripts` + табуляция + `rain.py`.

    Ошибка безобидна для кода и разрушительна для инструкции: пользователь
    копирует команду, и она не работает.
    """
    raw = io.open(path, encoding="utf-8", newline="").read()
    bad = {c for c in raw if ord(c) < 32 and c not in ("\n", "\r", "\t")}
    assert not bad, f"управляющие символы: {bad!r}"
    assert "\t" not in raw, "табуляция в тексте — обычно съеденный обратный слэш"
