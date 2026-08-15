"""Возврат моделей из **полных** прогонов в проект Unity.

Зачем нужен. Ноутбук и `scripts/train.py` пишут обученную модель по одному
и тому же пути `Assets/Envs/E##_<Name>/Models/*.onnx`. Поэтому запуск ноутбука
в режиме `QUICK_RUN = True` — а он предусмотрен требованием 9.3 и выполняется
при каждой проверке — затирает модель полного прогона моделью смоук-теста.
Файл при этом остаётся валидным и проходит верификацию, но агент в Unity
ведёт себя хуже, чем указано в карточке среды, и причину не видно.

Скрипт восстанавливает в проекте Unity модель **последнего полного** прогона
каждой пары (среда, алгоритм). Полным считается прогон, у которого в
`env_info.json` признак `quick` равен `false` — именно `false`, а не «ключа
нет»: каталог неизвестного происхождения источником модели не становится.

Использование::

    python scripts/sync_models.py            # показать, что будет сделано
    python scripts/sync_models.py --apply    # скопировать
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from labrl.utils.config import load_config, resolve_path  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"


def latest_full_run(env_id: str, algo: str) -> Path | None:
    """Каталог последнего полного прогона пары (среда, алгоритм)."""
    root = RESULTS / env_id / algo
    if not root.is_dir():
        return None

    for run in sorted(root.iterdir(), reverse=True):
        info_path = run / "env_info.json"
        onnx_path = run / "onnx" / "policy.onnx"
        if not info_path.is_file() or not onnx_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Прогон считается полным, только если он **явно** об этом заявил.
        # Отсутствие ключа — не «значит, полный»: так писали env_info.json
        # ноутбуки до того, как в них добавили запись признака, и один такой
        # каталог однажды подсунул в Unity модель смоук-теста
        # (docs/07_TROUBLESHOOTING.md, T-14). Умолчание выбрано в безопасную
        # сторону: неизвестное происхождение — не источник модели.
        if info.get("quick") is not False:
            continue
        return run
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="выполнить копирование")
    args = parser.parse_args()

    missing = 0
    for config_path in sorted(CONFIGS.glob("E*__*.yaml")):
        cfg = load_config(config_path)
        dest = resolve_path(cfg.export["onnx_path"])
        run = latest_full_run(cfg.env_id, cfg.algo_name)

        if run is None:
            print(f"[нет прогона] {cfg.env_id}/{cfg.algo_name}: полного прогона не найдено")
            missing += 1
            continue

        source = run / "onnx" / "policy.onnx"
        same = dest.is_file() and dest.read_bytes() == source.read_bytes()
        state = "совпадает" if same else "ОТЛИЧАЕТСЯ"
        print(f"[{state}] {cfg.env_id}/{cfg.algo_name}\n"
              f"    из  {source.relative_to(REPO_ROOT)}\n"
              f"    в   {dest.relative_to(REPO_ROOT)}")

        if args.apply and not same:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
            print("    скопировано")

    if not args.apply:
        print("\nэто был сухой прогон; для копирования добавьте --apply")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
