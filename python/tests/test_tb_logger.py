"""Логгер TensorBoard: полнота обязательной схемы 11.2 и структура прогона 11.1."""

from __future__ import annotations

from labrl.logging.run_dir import create_run_dir
from labrl.logging.tb_logger import REQUIRED_TAGS, TBLogger, Tags


def test_missing_required_tags_reports_gaps(tmp_path):
    with TBLogger(tmp_path / "tb") as log:
        log.scalar(Tags.CUMULATIVE_REWARD, 1.0, step=0)
        missing = log.missing_required_tags()

    assert Tags.CUMULATIVE_REWARD not in missing
    assert Tags.EVAL_SUCCESS_RATE in missing
    assert len(missing) == len(REQUIRED_TAGS) - 1


def test_complete_run_reports_no_gaps(tmp_path):
    with TBLogger(tmp_path / "tb") as log:
        for tag in REQUIRED_TAGS:
            log.scalar(tag, 0.0, step=0)
        assert log.missing_required_tags() == ()


def test_namespaces_are_applied(tmp_path):
    with TBLogger(tmp_path / "tb") as log:
        log.custom("TD Error", 0.5, step=1)
        log.env_stats({"SuccessRate": 1.0}, step=1)
        tags = log.seen_tags

    assert "Custom/TD Error" in tags
    assert "Env/SuccessRate" in tags


def test_metrics_json_written(tmp_path):
    import json

    with TBLogger(tmp_path / "tb") as log:
        log.scalar(Tags.EVAL_MEAN_REWARD, 0.91, step=100)
        path = log.dump_metrics(tmp_path / "metrics.json", extra={"seed": 0})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["last_values"][Tags.EVAL_MEAN_REWARD] == 0.91
    assert payload["seed"] == 0


def test_run_dir_layout_matches_specification(tmp_path):
    run = create_run_dir("E03_RollerBall", "dqn", seed=2, results_root=tmp_path)

    assert run.tb.is_dir() and run.ckpt.is_dir() and run.onnx.is_dir()
    assert run.root.name.endswith("_seed2")
    assert run.root.parent.name == "dqn"
    assert run.root.parent.parent.name == "E03_RollerBall"


def test_run_dir_writes_env_info(tmp_path):
    import json

    run = create_run_dir("E01_GridWorld", "qlearning", seed=0, results_root=tmp_path)
    run.write_env_info({"obs_shapes": [[25]], "mode": "editor"})

    assert json.loads(run.env_info_json.read_text(encoding="utf-8"))["mode"] == "editor"


# --- режим без записи на диск -------------------------------------------


def test_logger_without_directory_writes_nothing_but_counts_tags():
    """`TBLogger(None)` нужен подбору гиперпараметров: десятки коротких
    прогонов не должны засорять `results/`.

    Учёт тегов и последних значений при этом обязан продолжать работать —
    иначе `missing_required_tags` и `dump_metrics` теряют смысл.
    """
    logger = TBLogger(None)
    logger.scalar(Tags.CUMULATIVE_REWARD, 1.5, 10)
    logger.custom("Something", 0.25, 10)
    logger.text("note", "текст")
    logger.hparams({"lr": 1e-3}, {"final": 1.0})

    assert logger.log_dir is None
    assert Tags.CUMULATIVE_REWARD in logger.seen_tags
    assert Tags.CUMULATIVE_REWARD not in logger.missing_required_tags()
    logger.flush()
    logger.close()


def test_logger_without_directory_still_dumps_metrics(tmp_path):
    logger = TBLogger(None)
    logger.scalar(Tags.EVAL_MEAN_REWARD, 0.75, 1)
    path = logger.dump_metrics(tmp_path / "metrics.json", extra={"seed": 0})
    logger.close()

    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["last_values"][Tags.EVAL_MEAN_REWARD] == 0.75
    assert payload["seed"] == 0
