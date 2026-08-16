"""Сводка результатов по всем прогонам: IQM с доверительным интервалом (12.3).

Читает `results/<env>/<algo>/<прогон>/metrics.json`, группирует по среде
и алгоритму и печатает таблицу в формате `docs/RESULTS.md`.

Почему IQM, а не среднее. Распределение итоговых наград по сидам в RL обычно
имеет тяжёлые хвосты: один удачный или один провальный сид сдвигает среднее
сильнее, чем разница между алгоритмами. Interquartile mean устойчив к обоим
хвостам. Сравнение алгоритмов по одному прогону инструкцией запрещено (12.3).

Использование::

    python scripts/results.py                       # печать в stdout
    python scripts/results.py --write docs/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from labrl.eval.aggregate import interquartile_mean, stratified_bootstrap_iqm  # noqa: E402


def collect_runs(results_root: Path, include_quick: bool = False) -> dict[tuple[str, str], list[dict]]:
    """Собирает `metrics.json` прогонов, сгруппированные по (среда, алгоритм).

    По умолчанию **быстрые прогоны отбрасываются**. Это не косметика: смешивать
    в одной сводке полные прогоны и смоук-тесты нельзя — IQM тогда описывает
    неизвестно что. Признак берётся из `env_info.json`, который пишет
    `scripts/train.py` рядом с метриками.

    Каталоги отсортированы по времени, и при нескольких прогонах одного сида
    в сводку идёт **последний** — тот, чей результат актуален.
    """
    runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for metrics_path in sorted(results_root.glob("*/*/*/metrics.json")):
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"пропущен {metrics_path}: {exc}", file=sys.stderr)
            continue

        info_path = metrics_path.parent / "env_info.json"
        info = {}
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}

        if bool(info.get("quick", False)) and not include_quick:
            continue

        env_id = metrics_path.parents[2].name
        algo = metrics_path.parents[1].name
        data["_run_dir"] = str(metrics_path.parent.relative_to(REPO_ROOT))
        data["_total_steps"] = info.get("total_steps")
        data["_quick"] = bool(info.get("quick", False))
        runs[(env_id, algo)].append(data)
    return runs


def summarize(runs: list[dict]) -> dict:
    """Считает IQM по последней оценке каждого сида.

    Один сид даёт одно число — итоговую оценку. Стратифицированный бутстрэп
    требует минимум двух сидов; при меньшем числе возвращается только точечная
    оценка, и это явно помечается.
    """
    by_seed: dict[int, float] = {}
    success_by_seed: dict[int, float] = {}
    for run in runs:
        seed = int(run.get("seed", -1))
        history = run.get("eval_history") or []
        if not history:
            continue
        # Берётся ПОСЛЕДНЯЯ оценка прогона, а не лучшая: лучшая — это выбор
        # по тем же данным, на которых измеряют, и систематически завышает результат.
        by_seed[seed] = float(history[-1][1])
        # Доля успешных эпизодов той же оценки. Для сред без потолка награды
        # и для сред с формированием награды приёмка идёт именно по ней
        # (T-15 и T-17), поэтому сводка обязана показывать обе величины.
        if len(history[-1]) > 2:
            success_by_seed[seed] = float(history[-1][2])

    if not by_seed:
        return {"seeds": 0}

    seeds = sorted(by_seed)
    scores = np.array([by_seed[s] for s in seeds], dtype=np.float64)

    summary = {
        "seeds": len(seeds),
        "seed_list": seeds,
        "scores": scores.tolist(),
        "iqm": interquartile_mean(scores),
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "steps": runs[-1].get("_total_steps"),
        "wall_time": float(np.sum([r.get("wall_time_sec", 0.0) for r in runs])),
        "missing_tags": sorted({t for r in runs for t in r.get("missing_required_tags", [])}),
    }

    if len(success_by_seed) == len(seeds):
        success = np.array([success_by_seed[s] for s in seeds], dtype=np.float64)
        summary["success_scores"] = success.tolist()
        summary["success_iqm"] = interquartile_mean(success)

    if len(seeds) >= 2:
        ci = stratified_bootstrap_iqm(scores[:, None], resamples=10_000, seed=0)
        summary["ci_low"], summary["ci_high"] = ci.ci_low, ci.ci_high
    return summary


def render(runs: dict[tuple[str, str], list[dict]]) -> str:
    lines = [
        "# RESULTS — сводная таблица результатов",
        "",
        f"**Дата генерации:** {date.today().isoformat()}",
        "**Генератор:** `python scripts/results.py --write docs/RESULTS.md`",
        "",
        "В сводку входят только **полные** прогоны: смоук-тесты (`--quick`) отброшены,",
        "иначе IQM описывал бы смесь разных бюджетов. При нескольких прогонах одного",
        "сида берётся последний по времени.",
        "",
        "Итоговая метрика примера — **IQM** (interquartile mean) итоговых оценок",
        "по сидам с доверительным интервалом по стратифицированному бутстрэпу",
        "(требование 12.3). Берётся **последняя** оценка каждого прогона, а не лучшая:",
        "выбор лучшей по тем же данным, на которых измеряют, систематически завышает результат.",
        "",
        "Колонка **IQM награды** сопоставима между средами только по смыслу самой",
        "среды: у награды `E05_FoodCollector` нет потолка, а награда `E06_Hunter3D`",
        "содержит смещение формирования. Там, где приёмка задана по исходу эпизода,",
        "смотреть надо на колонку **IQM успеха** (T-15 и T-17 в",
        "`docs/07_TROUBLESHOOTING.md`).",
        "",
        "| Среда | Алгоритм | Сидов | IQM награды | 95% CI | IQM успеха | Разброс по сидам | Время, с |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for (env_id, algo) in sorted(runs):
        s = summarize(runs[(env_id, algo)])
        if not s["seeds"]:
            lines.append(f"| `{env_id}` | `{algo}` | 0 | — | — | — | нет завершённых оценок | — |")
            continue
        ci = (f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
              if "ci_low" in s else "нужно ≥ 2 сидов")
        success = f"{s['success_iqm']:.4f}" if "success_iqm" in s else "—"
        lines.append(
            f"| `{env_id}` | `{algo}` | {s['seeds']} | {s['iqm']:.4f} | {ci} | {success} | "
            f"[{s['min']:.4f}, {s['max']:.4f}] | {s['wall_time']:.0f} |"
        )

    lines += ["", "## Подробности по прогонам", ""]
    for (env_id, algo) in sorted(runs):
        s = summarize(runs[(env_id, algo)])
        lines.append(f"### `{env_id}` / `{algo}`")
        lines.append("")
        if not s["seeds"]:
            lines += ["Завершённых оценок нет.", ""]
            continue
        lines.append(f"- сиды: {s['seed_list']}, бюджет шагов: {s['steps']}")
        lines.append(f"- итоговые оценки: {[round(x, 4) for x in s['scores']]}")
        lines.append(f"- IQM награды: **{s['iqm']:.4f}**, среднее {s['mean']:.4f}")
        if "success_iqm" in s:
            lines.append(f"- доля успешных эпизодов по сидам: "
                         f"{[round(x, 4) for x in s['success_scores']]}, "
                         f"IQM **{s['success_iqm']:.4f}**")
        if s["missing_tags"]:
            lines.append(f"- **обязательные теги схемы 11.2 без данных: {s['missing_tags']}**")
        else:
            lines.append("- все обязательные теги схемы 11.2 присутствуют")
        lines.append("")
        for run in sorted(runs[(env_id, algo)], key=lambda r: r.get("seed", -1)):
            history = run.get("eval_history") or []
            tail = ", ".join(f"{step}: {reward:+.4f}" for step, reward, _ in history[-4:])
            lines.append(f"  - сид {run.get('seed')}: `{run['_run_dir']}` — последние оценки: {tail}")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(REPO_ROOT / "results"))
    parser.add_argument("--write", help="куда записать Markdown; без него — печать в stdout")
    parser.add_argument("--include-quick", action="store_true",
                        help="включить смоук-прогоны (--quick); по умолчанию отбрасываются")
    args = parser.parse_args()

    results_root = Path(args.results)
    if not results_root.is_dir():
        print(f"каталог результатов не найден: {results_root}", file=sys.stderr)
        return 1

    runs = collect_runs(results_root, include_quick=args.include_quick)
    if not runs:
        print("прогонов не найдено", file=sys.stderr)
        return 1

    text = render(runs)
    if args.write:
        dest = Path(args.write)
        if not dest.is_absolute():
            dest = REPO_ROOT / dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8", newline="\n")
        print(f"записано: {dest}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
