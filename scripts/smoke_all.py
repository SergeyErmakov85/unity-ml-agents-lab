"""Смоук-проверка всех примеров одной командой (DoD раздела 15 инструкции).

Что делает: для каждого конфига в `configs/` запускает `scripts/train.py`
с флагом `--quick` и собирает итог в таблицу. Ничего не обучает по-настоящему:
бюджет сокращён вдесятеро, критерий приёмки не проверяется. Проверяется
**конвейер** — что среда открывается, алгоритм собирается, обучение идёт,
экспорт в ONNX проходит верификацию, обязательные теги схемы 11.2 пишутся.

Использование::

    python scripts/smoke_all.py                      # все конфиги
    python scripts/smoke_all.py --env E08 E09        # только эти среды
    python scripts/smoke_all.py --skip E07           # кроме этих
    python scripts/smoke_all.py --list               # что будет запущено

Почему отдельным процессом на конфиг, а не импортом в цикле. Три причины,
и все три встретились на практике:

* импорт `mlagents.torch_utils` глобально переключает устройство по умолчанию
  (`torch.set_default_device`), и это протекает между прогонами;
* Unity-билд держит порт; падение одного прогона не должно уносить
  остальные;
* утечка памяти или зависший процесс среды остаются внутри своего процесса.

Каждый прогон получает **свой** ``--worker-id``, поэтому при желании их
можно пустить параллельно; здесь они идут последовательно, чтобы не
конкурировать за GPU и не путать вывод.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from labrl.utils.config import load_config  # noqa: E402

#: Сколько ждать один смоук-прогон, секунд. Самый долгий из измеренных —
#: `E11_Research__fca_ppo` (257 с) плюс запас на сборку окружения.
TIMEOUT_SECONDS = 1800


def python_executable() -> str:
    """Интерпретатор из `python/.venv`, а не тот, которым запущен скрипт.

    Разница существенна: смоук-проверку удобно запускать из любой оболочки,
    а стек RL живёт только в venv лаборатории.
    """
    venv = REPO_ROOT / "python" / ".venv" / "Scripts" / "python.exe"
    return str(venv if venv.is_file() else sys.executable)


def collect_configs(only: list[str], skip: list[str]) -> list[Path]:
    """Конфиги в порядке возрастания кода примера."""
    paths = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    if only:
        paths = [p for p in paths if any(p.stem.startswith(prefix) for prefix in only)]
    if skip:
        paths = [p for p in paths if not any(p.stem.startswith(prefix) for prefix in skip)]
    return paths


def build_exists(config_path: Path) -> bool:
    cfg = load_config(config_path)
    if cfg.env.get("mode") != "build":
        return True
    return (REPO_ROOT / str(cfg.env["build_path"])).is_file()


def run_one(config_path: Path, worker_id: int) -> dict:
    """Один смоук-прогон в отдельном процессе."""
    started = time.perf_counter()
    completed = subprocess.run(
        [
            python_executable(), str(REPO_ROOT / "scripts" / "train.py"),
            "--config", str(config_path),
            "--seed", "0",
            "--quick",
            "--worker-id", str(worker_id),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_SECONDS,
    )
    elapsed = time.perf_counter() - started

    # Сводка прогона печатается как JSON последним блоком; вытаскиваем её,
    # чтобы показать долю успехов, а не только код возврата.
    summary: dict = {}
    text = completed.stdout or ""
    if "{" in text:
        start = text.rindex("{")
        try:
            summary = json.loads(text[start : text.rindex("}") + 1])
        except (ValueError, IndexError):
            summary = {}

    return {
        "config": config_path.stem,
        "returncode": completed.returncode,
        "seconds": elapsed,
        "eval_success": summary.get("eval_success"),
        "missing_tags": summary.get("missing_required_tags", []),
        "stderr_tail": (completed.stderr or "").strip().splitlines()[-3:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", nargs="*", default=[],
                        help="префиксы имён конфигов, например E08 E09")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="префиксы, которые пропустить")
    parser.add_argument("--list", action="store_true",
                        help="показать, что будет запущено, и выйти")
    args = parser.parse_args()

    configs = collect_configs(args.env, args.skip)
    if not configs:
        print("нечего запускать: под фильтр не попал ни один конфиг", flush=True)
        return 1

    if args.list:
        for path in configs:
            mark = "" if build_exists(path) else "   (нет билда)"
            print(f"  {path.stem}{mark}")
        return 0

    missing_builds = [p.stem for p in configs if not build_exists(p)]
    if missing_builds:
        print("Нет билдов для:", ", ".join(missing_builds), flush=True)
        print("Соберите их: python scripts/build_env.py <env_id>", flush=True)
        print(flush=True)

    results = []
    for index, path in enumerate(configs):
        if not build_exists(path):
            results.append({"config": path.stem, "returncode": None, "seconds": 0.0,
                            "eval_success": None, "missing_tags": [], "stderr_tail": []})
            continue

        print(f"[{index + 1}/{len(configs)}] {path.stem} …", flush=True)
        try:
            outcome = run_one(path, worker_id=index % 8)
        except subprocess.TimeoutExpired:
            outcome = {"config": path.stem, "returncode": -1, "seconds": TIMEOUT_SECONDS,
                       "eval_success": None, "missing_tags": [],
                       "stderr_tail": [f"превышен лимит {TIMEOUT_SECONDS} с"]}
        results.append(outcome)

        status = "ок" if outcome["returncode"] == 0 else f"ОШИБКА ({outcome['returncode']})"
        print(f"      {status}, {outcome['seconds']:.0f} с", flush=True)
        for line in outcome["stderr_tail"] if outcome["returncode"] not in (0, None) else []:
            print(f"      {line}", flush=True)

    # --- сводка ---------------------------------------------------------
    print()
    print(f"{'конфиг':<40} {'итог':>10} {'время, с':>10} {'успех':>8}  теги")
    print("-" * 86)
    for item in results:
        if item["returncode"] is None:
            verdict = "нет билда"
        elif item["returncode"] == 0:
            verdict = "ок"
        else:
            verdict = "ОШИБКА"
        success = "—" if item["eval_success"] is None else f"{item['eval_success']:.0%}"
        tags = "все" if not item["missing_tags"] else f"НЕТ: {len(item['missing_tags'])}"
        print(f"{item['config']:<40} {verdict:>10} {item['seconds']:>10.0f} {success:>8}  {tags}")

    failed = [r for r in results if r["returncode"] not in (0, None)]
    skipped = [r for r in results if r["returncode"] is None]
    print()
    print(f"прогонов: {len(results)}, успешно: {len(results) - len(failed) - len(skipped)}, "
          f"с ошибкой: {len(failed)}, пропущено (нет билда): {len(skipped)}")
    print("напоминание: это смоук-тест конвейера — критерий приёмки здесь "
          "не проверяется (бюджет сокращён вдесятеро)")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
