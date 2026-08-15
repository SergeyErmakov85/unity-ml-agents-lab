"""Проверка ONNX-модели по контракту ML-Agents из CLI (требование 10.5).

Полная верификация требует того же модуля политики, что экспортировался
(для числового паритета), поэтому она живёт в ноутбуке и в `scripts/train.py`.
Этот скрипт делает то, что можно сделать по одному лишь файлу: структура,
opset, имена входов и выходов, значения констант, прогон на батчах 1 и 64,
диапазоны действий.

Использование::

    python scripts/verify_onnx.py path/to/model.onnx --discrete-branches 4
    python scripts/verify_onnx.py path/to/model.onnx --continuous-size 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from labrl.export.onnx_export import (  # noqa: E402
    MODEL_EXPORT_VERSION,
    ONNX_OPSET,
    ActionSpecLite,
    contract_input_names,
    contract_output_names,
)
from labrl.export.onnx_verify import VerificationResult  # noqa: E402


def obs_shapes_from_model(model: onnx.ModelProto) -> list[tuple[int, ...]]:
    """Восстанавливает формы наблюдений из самого графа.

    Батч (нулевая ось) отбрасывается: он динамический и в контракте помечен
    как `batch`.
    """
    shapes: dict[int, tuple[int, ...]] = {}
    for inp in model.graph.input:
        if not inp.name.startswith("obs_"):
            continue
        dims = inp.type.tensor_type.shape.dim
        shapes[int(inp.name.removeprefix("obs_"))] = tuple(d.dim_value for d in dims[1:])
    return [shapes[i] for i in sorted(shapes)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--discrete-branches", type=int, nargs="*", default=[],
                        help="размеры дискретных веток, например --discrete-branches 4")
    parser.add_argument("--continuous-size", type=int, default=0)
    parser.add_argument("--memory-size", type=int, default=0)
    args = parser.parse_args()

    spec = ActionSpecLite(
        continuous_size=args.continuous_size,
        discrete_branches=tuple(args.discrete_branches),
    )

    result = VerificationResult()
    model = onnx.load(str(args.onnx_path))

    try:
        onnx.checker.check_model(model)
        result.add("структура (onnx.checker)", True)
    except Exception as exc:  # noqa: BLE001
        result.add("структура (onnx.checker)", False, str(exc))

    opsets = {imp.domain: imp.version for imp in model.opset_import}
    actual_opset = opsets.get("", opsets.get("ai.onnx"))
    result.add("opset", actual_opset == ONNX_OPSET, f"ожидался {ONNX_OPSET}, в модели {actual_opset}")

    obs_shapes = obs_shapes_from_model(model)
    result.add("наблюдения найдены", len(obs_shapes) > 0, f"формы: {obs_shapes}")

    expected_inputs = set(contract_input_names(len(obs_shapes)))
    expected_outputs = set(contract_output_names(spec, args.memory_size))
    actual_inputs = {i.name for i in model.graph.input}
    actual_outputs = {o.name for o in model.graph.output}
    result.add("имена входов", actual_inputs == expected_inputs,
               f"лишние={sorted(actual_inputs - expected_inputs)}, отсутствуют={sorted(expected_inputs - actual_inputs)}")
    result.add("имена выходов", actual_outputs == expected_outputs,
               f"лишние={sorted(actual_outputs - expected_outputs)}, отсутствуют={sorted(expected_outputs - actual_outputs)}")

    session = ort.InferenceSession(str(args.onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]
    mask_size = int(sum(spec.discrete_branches))
    rng = np.random.default_rng(0)

    for batch in (1, 64):
        feed = {f"obs_{i}": rng.standard_normal((batch, *shape)).astype(np.float32)
                for i, shape in enumerate(obs_shapes)}
        feed["action_masks"] = np.ones((batch, mask_size), dtype=np.float32)
        feed["recurrent_in"] = np.zeros((batch, 1, args.memory_size), dtype=np.float32)
        try:
            values = dict(zip(out_names, session.run(out_names, feed)))
            result.add(f"прогон onnxruntime, батч {batch}", True)
        except Exception as exc:  # noqa: BLE001
            result.add(f"прогон onnxruntime, батч {batch}", False, str(exc))
            continue

        if batch == 1:
            _check_const(result, values, "version_number", [float(MODEL_EXPORT_VERSION)])
            _check_const(result, values, "memory_size", [float(args.memory_size)])
            if spec.continuous_size > 0:
                _check_const(result, values, "continuous_action_output_shape", [float(spec.continuous_size)])
            if spec.discrete_branches:
                _check_const(result, values, "discrete_action_output_shape",
                             [float(b) for b in spec.discrete_branches])
        else:
            if spec.continuous_size > 0:
                v = values["continuous_actions"]
                result.add("диапазон непрерывных действий", bool(v.min() >= -1 and v.max() <= 1),
                           f"[{v.min():.4f}, {v.max():.4f}]")
            for idx, size in enumerate(spec.discrete_branches):
                v = values["discrete_actions"][:, idx]
                ok = bool(np.all(v == np.floor(v)) and np.all((v >= 0) & (v < size)))
                result.add(f"диапазон дискретной ветки {idx}", ok, f"[{v.min():.0f}, {v.max():.0f}], ожидалось [0, {size})")

    print(result.report())
    print("\nЧисловой паритет с PyTorch по одному файлу проверить нельзя — "
          "нужен исходный модуль политики. См. labrl.export.onnx_verify.verify_onnx_model.")
    return 0 if result.passed else 1


def _check_const(result: VerificationResult, values: dict, name: str, expected: list[float]) -> None:
    if name not in values:
        result.add(f"константа {name}", False, "выход отсутствует")
        return
    actual = np.asarray(values[name], dtype=np.float64).reshape(-1).tolist()
    ok = len(actual) == len(expected) and all(abs(a - e) < 1e-6 for a, e in zip(actual, expected))
    result.add(f"константа {name}", ok, f"ожидалось {expected}, получено {actual}")


if __name__ == "__main__":
    sys.exit(main())
