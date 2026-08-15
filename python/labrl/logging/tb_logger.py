"""Логирование в TensorBoard по обязательной схеме раздела 11 инструкции.

Имена тегов вынесены в константы: одинаковые имена во всех примерах — это то,
что позволяет сравнивать прогоны разных алгоритмов на одном графике и совпадает
с тегами штатного тренера ML-Agents.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from torch.utils.tensorboard import SummaryWriter


class Tags:
    """Обязательные скаляры (таблица 11.2) и неймспейсы (11.3–11.4)."""

    CUMULATIVE_REWARD = "Environment/Cumulative Reward"
    EPISODE_LENGTH = "Environment/Episode Length"
    VALUE_LOSS = "Losses/Value Loss"
    POLICY_LOSS = "Losses/Policy Loss"
    ENTROPY = "Policy/Entropy"
    LEARNING_RATE = "Policy/Learning Rate"
    EPSILON = "Policy/Epsilon"
    EVAL_MEAN_REWARD = "Eval/Mean Reward"
    EVAL_SUCCESS_RATE = "Eval/Success Rate"
    STEPS_PER_SECOND = "Perf/Steps Per Second"

    #: Метрики, специфичные для алгоритма (11.3).
    CUSTOM_NS = "Custom/"
    #: Метрики, пришедшие из Unity через StatsSideChannel (11.4).
    ENV_NS = "Env/"


#: Полный список обязательных тегов — используется тестом покрытия схемы.
REQUIRED_TAGS: tuple[str, ...] = (
    Tags.CUMULATIVE_REWARD,
    Tags.EPISODE_LENGTH,
    Tags.VALUE_LOSS,
    Tags.POLICY_LOSS,
    Tags.ENTROPY,
    Tags.LEARNING_RATE,
    Tags.EPSILON,
    Tags.EVAL_MEAN_REWARD,
    Tags.EVAL_SUCCESS_RATE,
    Tags.STEPS_PER_SECOND,
)


def git_commit_hash() -> str:
    """Хэш текущего коммита (требование 11.5). ``unknown`` вне git-репозитория."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).resolve().parent),
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


class TBLogger:
    """Тонкая обёртка над ``SummaryWriter``.

    Обёртка нужна ровно для трёх вещей: единые имена тегов, запись всех
    записанных тегов (для проверки полноты схемы 11.2) и запись сводки прогона
    в ``metrics.json``.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self._seen_tags: set[str] = set()
        self._last: dict[str, float] = {}

    # --- запись ---------------------------------------------------------

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, float(value), global_step=step)
        self._seen_tags.add(tag)
        self._last[tag] = float(value)

    def scalars(self, values: Mapping[str, float], step: int, prefix: str = "") -> None:
        for tag, value in values.items():
            self.scalar(prefix + tag, value, step)

    def custom(self, name: str, value: float, step: int) -> None:
        """Метрика алгоритма: попадает в неймспейс ``Custom/`` (11.3)."""
        self.scalar(Tags.CUSTOM_NS + name, value, step)

    def env_stats(self, values: Mapping[str, float], step: int) -> None:
        """Статистики из Unity: неймспейс ``Env/`` (11.4)."""
        self.scalars(values, step, prefix=Tags.ENV_NS)

    def text(self, tag: str, body: str, step: int = 0) -> None:
        self._writer.add_text(tag, body, global_step=step)

    def hparams(self, hparams: Mapping[str, Any], metrics: Mapping[str, float]) -> None:
        """Гиперпараметры прогона (требование 11.5).

        ``add_hparams`` принимает только скаляры и строки, поэтому вложенные
        структуры сериализуются в JSON-строку.
        """
        flat = {k: (v if isinstance(v, (int, float, str, bool)) else json.dumps(v, ensure_ascii=False))
                for k, v in hparams.items()}
        self._writer.add_hparams(flat, dict(metrics))

    # --- завершение -----------------------------------------------------

    @property
    def seen_tags(self) -> frozenset[str]:
        """Теги, реально записанные в этот прогон."""
        return frozenset(self._seen_tags)

    def missing_required_tags(self) -> tuple[str, ...]:
        """Обязательные теги 11.2, которых в прогоне нет."""
        return tuple(t for t in REQUIRED_TAGS if t not in self._seen_tags)

    def dump_metrics(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> Path:
        """Пишет ``metrics.json`` — последние значения всех тегов прогона."""
        out = Path(path)
        payload: dict[str, Any] = {"last_values": self._last, "tags": sorted(self._seen_tags)}
        if extra:
            payload.update(extra)
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return out

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> "TBLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
