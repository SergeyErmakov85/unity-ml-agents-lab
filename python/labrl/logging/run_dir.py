"""Каталог прогона обучения (структура 11.1).

::

    results/E##_<Name>/<algo>/<YYYYMMDD-HHMMSS>_seed<k>/
        tb/              логи TensorBoard
        ckpt/            чекпойнты
        onnx/            экспортированные модели
        config.yaml      копия конфига эксперимента
        env_info.json    что за среда, размерности пространств, режим подключения
        metrics.json     сводка метрик прогона
        pip_freeze.txt   слепок окружения
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from labrl.utils.config import repo_root


@dataclass(frozen=True)
class RunDir:
    """Пути внутри каталога прогона. Все подкаталоги создаются при создании."""

    root: Path

    @property
    def tb(self) -> Path:
        return self.root / "tb"

    @property
    def ckpt(self) -> Path:
        return self.root / "ckpt"

    @property
    def onnx(self) -> Path:
        return self.root / "onnx"

    @property
    def config_yaml(self) -> Path:
        return self.root / "config.yaml"

    @property
    def env_info_json(self) -> Path:
        return self.root / "env_info.json"

    @property
    def metrics_json(self) -> Path:
        return self.root / "metrics.json"

    @property
    def pip_freeze_txt(self) -> Path:
        return self.root / "pip_freeze.txt"

    def write_env_info(self, info: Mapping[str, Any]) -> Path:
        with self.env_info_json.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(dict(info), fh, ensure_ascii=False, indent=2)
        return self.env_info_json

    def write_pip_freeze(self) -> Path:
        """Слепок окружения текущего интерпретатора."""
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, timeout=120,
            )
            body = out.stdout if out.returncode == 0 else f"pip freeze завершился с кодом {out.returncode}\n{out.stderr}"
        except (OSError, subprocess.SubprocessError) as exc:
            body = f"pip freeze не выполнен: {exc}"
        self.pip_freeze_txt.write_text(body, encoding="utf-8", newline="\n")
        return self.pip_freeze_txt


def create_run_dir(env_id: str, algo: str, seed: int, results_root: str | Path | None = None) -> RunDir:
    """Создаёт каталог прогона по схеме 11.1 и возвращает его пути.

    Args:
        env_id: идентификатор среды, например ``E03_RollerBall``.
        algo: имя алгоритма в нижнем регистре, например ``dqn``.
        seed: сид прогона.
        results_root: корень результатов; по умолчанию ``<repo>/results``.
    """
    root = Path(results_root) if results_root is not None else repo_root() / "results"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = RunDir(root / env_id / algo / f"{stamp}_seed{seed}")
    for d in (run.root, run.tb, run.ckpt, run.onnx):
        d.mkdir(parents=True, exist_ok=True)
    return run
