"""Headless-сборка среды из CLI (требование 7.6).

Скрипт — тонкая обёртка над `Unity.exe -batchmode`: он находит редактор нужной
версии, собирает командную строку и **прозрачно пробрасывает код возврата**.
Последнее и есть смысл обёртки: забыть проверить `$LASTEXITCODE` легко,
а провалившаяся сборка, посчитанная успешной, стоит часа отладки.

Использование::

    python scripts/build_env.py E03_RollerBall
    python scripts/build_env.py E03_RollerBall --output builds/E03_RollerBall --log build.log
    python scripts/build_env.py --validate-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UNITY_PROJECT = REPO_ROOT / "unity" / "MLAgentsLab"
PROJECT_VERSION_FILE = UNITY_PROJECT / "ProjectSettings" / "ProjectVersion.txt"
HUB_EDITOR_ROOT = Path(r"C:\Program Files\Unity\Hub\Editor")

BUILD_METHOD = "LabRL.EditorTools.BuildScript.BuildEnv"
VALIDATE_METHOD = "LabRL.EditorTools.SceneValidator.ValidateAllBatch"


def project_unity_version() -> str:
    """Версия редактора, которой создан проект. Источник — ProjectVersion.txt."""
    for line in PROJECT_VERSION_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("m_EditorVersion:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"в {PROJECT_VERSION_FILE} нет строки m_EditorVersion")


def find_unity(version: str | None = None) -> Path:
    """Находит Unity.exe нужной версии.

    Открывать проект редактором другой версии нельзя: Unity молча обновит
    ProjectSettings и Library, и проект перестанет открываться заявленной
    версией. Поэтому несовпадение — ошибка, а не предупреждение.
    """
    version = version or project_unity_version()
    exe = HUB_EDITOR_ROOT / version / "Editor" / "Unity.exe"
    if exe.is_file():
        return exe

    installed = sorted(p.name for p in HUB_EDITOR_ROOT.iterdir()) if HUB_EDITOR_ROOT.is_dir() else []
    raise SystemExit(
        f"Unity {version} не найден: ожидался {exe}\n"
        f"Установлено в Hub: {', '.join(installed) or '<ничего>'}\n"
        f"Проект требует версию из {PROJECT_VERSION_FILE.relative_to(REPO_ROOT)}."
    )


def run_unity(method: str, extra_args: list[str], log_path: Path) -> int:
    """Запускает Unity в batch-режиме и возвращает код завершения."""
    unity = find_unity()
    cmd = [
        str(unity), "-batchmode", "-quit", "-nographics",
        "-projectPath", str(UNITY_PROJECT),
        "-executeMethod", method,
        "-logFile", str(log_path),
        *extra_args,
    ]
    print("Запуск:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd)
    print(f"Unity завершился с кодом {completed.returncode}; лог: {log_path}", flush=True)
    if completed.returncode != 0 and log_path.is_file():
        print("--- последние 40 строк лога ---")
        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env_id", nargs="?", help="идентификатор среды, например E03_RollerBall")
    parser.add_argument("--output", help="каталог сборки; по умолчанию builds/<env_id>")
    parser.add_argument("--log", default="build.log", help="файл лога Unity")
    parser.add_argument("--validate-only", action="store_true",
                        help="только прогнать SceneValidator по всем средам, не собирать")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = REPO_ROOT / log_path

    if args.validate_only:
        return run_unity(VALIDATE_METHOD, [], log_path)

    if not args.env_id:
        parser.error("укажите env_id либо --validate-only")

    output = args.output or f"builds/{args.env_id}"
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    return run_unity(BUILD_METHOD, ["-envId", args.env_id, "-outputPath", str(output_path)], log_path)


if __name__ == "__main__":
    sys.exit(main())
