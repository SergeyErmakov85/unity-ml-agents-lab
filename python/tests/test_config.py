"""Конфиг эксперимента: обязательные блоки и схема идентификатора среды."""

from __future__ import annotations

import copy

import pytest
import yaml

from labrl.utils.config import ExperimentConfig, dump_config, load_config, repo_root

VALID = {
    "experiment": {"env_id": "E03_RollerBall", "algo": "dqn", "seeds": [0, 1, 2], "total_steps": 1000},
    "env": {"mode": "build", "build_path": "builds/E03_RollerBall/E03_RollerBall.exe"},
    "network": {"type": "mlp", "hidden_sizes": [128, 128]},
    "algo": {"gamma": 0.99, "lr": 3e-4},
    "eval": {"every_steps": 100, "episodes": 5, "deterministic": True},
    "success_criteria": {"metric": "Eval/Mean Reward", "threshold": 0.85, "within_steps": 1000},
    "export": {"onnx_path": "unity/x.onnx", "verify": True},
}


def test_valid_config_parses():
    cfg = ExperimentConfig(raw=copy.deepcopy(VALID))
    assert cfg.env_id == "E03_RollerBall"
    assert cfg.algo_name == "dqn"
    assert cfg.seeds == [0, 1, 2]
    assert cfg.total_steps == 1000
    assert cfg.success_criteria.threshold == pytest.approx(0.85)


@pytest.mark.parametrize("section", list(VALID))
def test_missing_section_rejected(section):
    raw = copy.deepcopy(VALID)
    del raw[section]
    with pytest.raises(ValueError, match="отсутствуют обязательные блоки"):
        ExperimentConfig(raw=raw)


@pytest.mark.parametrize(
    "env_id",
    ["RollerBall", "E3_RollerBall", "E03RollerBall", "E03_rollerBall", "E03_", "E003_X", 42],
)
def test_invalid_env_id_rejected(env_id):
    """env_id — Behavior Name в Unity (5.3); опечатка здесь ломает связь с Unity молча."""
    raw = copy.deepcopy(VALID)
    raw["experiment"]["env_id"] = env_id
    with pytest.raises(ValueError, match="E##"):
        ExperimentConfig(raw=raw)


@pytest.mark.parametrize("env_id", ["E00_Bandit", "E01_GridWorld", "E11_Research"])
def test_valid_env_ids_accepted(env_id):
    raw = copy.deepcopy(VALID)
    raw["experiment"]["env_id"] = env_id
    assert ExperimentConfig(raw=raw).env_id == env_id


def test_roundtrip_through_yaml(tmp_path):
    src = tmp_path / "cfg.yaml"
    src.write_text(yaml.safe_dump(VALID, allow_unicode=True), encoding="utf-8")

    cfg = load_config(src)
    dest = dump_config(cfg, tmp_path / "copy.yaml")

    assert yaml.safe_load(dest.read_text(encoding="utf-8")) == VALID


def test_repo_root_points_at_repository():
    """Пути в конфигах относительны корню репозитория, а не текущему каталогу (9.5)."""
    root = repo_root()
    assert (root / "python" / "labrl").is_dir()
    assert (root / "unity" / "MLAgentsLab").is_dir()
