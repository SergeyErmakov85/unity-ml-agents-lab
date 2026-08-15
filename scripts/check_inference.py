"""Проверка инференса ONNX **в Unity** и сравнение с Python-оценкой (требование 10.6).

Python-верификация ONNX доказывает, что файл соответствует контракту и численно
совпадает с PyTorch. Она **не** доказывает, что агент в Unity ведёт себя так же:
разойтись могут порядок наблюдений, нормализация, частота принятия решений.
Этот скрипт закрывает разрыв: собирает билд с назначенной моделью, прогоняет
20 эпизодов и сравнивает среднюю награду с Python-оценкой.

Критерий приёмки: награда в Unity ≥ **0.8 ×** награды в Python.

Использование::

    python scripts/check_inference.py --config configs/E01_GridWorld__qlearning.yaml \\
        --python-reward 0.68
    python scripts/check_inference.py --config ... --skip-build   # переиспользовать билд
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from labrl.utils.config import load_config, resolve_path  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_env import find_unity  # noqa: E402

#: Порог приёмки из требования 10.6.
ACCEPTANCE_RATIO = 0.8

BUILD_METHOD = "LabRL.EditorTools.InferenceBuild.Build"


def build_inference(env_id: str, model_path: Path, output_dir: Path, log_path: Path) -> int:
    """Собирает билд с назначенной моделью и `Inference Only`."""
    unity = find_unity()
    project = REPO_ROOT / "unity" / "MLAgentsLab"
    model_rel = model_path.relative_to(project).as_posix()

    cmd = [
        str(unity), "-batchmode", "-quit", "-nographics",
        "-projectPath", str(project),
        "-executeMethod", BUILD_METHOD,
        "-envId", env_id,
        "-modelPath", model_rel,
        "-outputPath", str(output_dir),
        "-logFile", str(log_path),
    ]
    print("Сборка инференс-билда:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd)
    if completed.returncode != 0 and log_path.is_file():
        print("--- последние 30 строк лога ---")
        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
    return completed.returncode


def run_inference(exe: Path, episodes: int, result_path: Path, log_path: Path, timeout_s: int) -> int:
    """Прогоняет билд и ждёт, пока проб запишет результат."""
    cmd = [
        str(exe), "-batchmode", "-nographics",
        "-logFile", str(log_path),
        "-inferenceEpisodes", str(episodes),
        "-inferenceResult", str(result_path),
        "-inferenceTimeout", str(timeout_s),
    ]
    print("Прогон инференса:", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python-reward", type=float, required=True,
                        help="средняя награда Python-оценки, с которой сравниваем")
    parser.add_argument("--episodes", type=int, default=20, help="требование 10.6 — 20 эпизодов")
    parser.add_argument("--timeout", type=int, default=300, help="секунд на прогон")
    parser.add_argument("--skip-build", action="store_true", help="использовать существующий билд")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_id = cfg.env_id
    model_path = resolve_path(cfg.export["onnx_path"])
    if not model_path.is_file():
        print(f"модель не найдена: {model_path}\nСначала обучите агента: "
              f"python scripts/train.py --config {args.config}", file=sys.stderr)
        return 1

    output_dir = REPO_ROOT / "builds" / f"{env_id}_inference"
    exe = output_dir / f"{env_id}_inference.exe"
    result_path = REPO_ROOT / "results" / env_id / "inference_check.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        code = build_inference(env_id, model_path, output_dir, REPO_ROOT / "logs-infbuild.log")
        if code != 0:
            print(f"сборка инференс-билда провалилась, код {code}", file=sys.stderr)
            return code

    if not exe.is_file():
        print(f"билд не найден: {exe}", file=sys.stderr)
        return 1

    if result_path.exists():
        result_path.unlink()

    code = run_inference(exe, args.episodes, result_path, REPO_ROOT / "logs-inference-player.log", args.timeout)
    if not result_path.is_file():
        print(f"проб не записал результат (код игрока {code})", file=sys.stderr)
        return 1

    data = json.loads(result_path.read_text(encoding="utf-8"))
    unity_reward = float(data["mean_reward"])
    ratio = unity_reward / args.python_reward if args.python_reward else float("inf")
    passed = data["episodes"] >= args.episodes and not data["timed_out"] and ratio >= ACCEPTANCE_RATIO

    print("\n=== Проверка инференса в Unity (требование 10.6) ===")
    print(f"  среда:            {env_id}")
    print(f"  модель:           {model_path.relative_to(REPO_ROOT)}")
    print(f"  эпизодов:         {data['episodes']} (запрошено {data['requested']})")
    print(f"  исходы:           {data['outcomes']}")
    print(f"  награда в Unity:  {unity_reward:.4f}  [{data['min_reward']:.4f}, {data['max_reward']:.4f}]")
    print(f"  награда в Python: {args.python_reward:.4f}")
    print(f"  отношение:        {ratio:.3f} (порог {ACCEPTANCE_RATIO})")
    print(f"  ИТОГ: {'ПРОЙДЕНО' if passed else 'ПРОВАЛЕНО'}")

    if not passed:
        print("\nТиповые причины расхождения (10.6): рассогласование порядка наблюдений, "
              "отсутствующая нормализация, различие в частоте принятия решений. "
              "Разбор — docs/07_TROUBLESHOOTING.md.", file=sys.stderr)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
