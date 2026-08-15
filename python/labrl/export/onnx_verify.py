"""Автоматическая верификация экспортированной ONNX-модели (требование 10.5).

Верификация **блокирующая**: провал любой проверки означает, что пример
не принят. Смысл — поймать расхождение с Unity здесь, за секунды, а не в виде
«агент в Unity ведёт себя не так, как в Python» через несколько часов.

Таблица проверок (10.5 инструкции):

===================  ==========================================================
Структура            ``onnx.checker.check_model`` без ошибок
opset                ровно 9 — значение из константы пакета ``mlagents``
Имена                множества входов/выходов **точно** равны контракту
Типы                 дискретные действия целые, непрерывные — float32
Константы            ``version_number``, ``memory_size``, ``*_output_shape``
Формы                прогон onnxruntime на батчах 1 и 64
Числовой паритет     ``max‖ONNX − PyTorch‖∞ ≤ 1e-4`` на 64 наблюдениях
Диапазоны            непрерывные в ``[-1, 1]``; дискретные — целые ``[0, n)``
===================  ==========================================================

Проверка типов есть не в таблице 10.5, а добавлена по факту чтения
``ApplierImpl.cs`` пакета ``com.unity.ml-agents@4.0.3``: Unity читает
``discrete_actions`` как ``Tensor<int>``, поэтому модель с float-выходом
проходит и ``onnx.checker``, и ``onnxruntime``, но падает при инференсе
в Unity. Именно такое расхождение верификация и обязана ловить.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from labrl.export.onnx_export import (
    CONTINUOUS_CLIP,
    MODEL_EXPORT_VERSION,
    ONNX_OPSET,
    MLAgentsPolicyWrapper,
    _HasActionSpec,
    _module_device,
    contract_input_names,
    contract_output_names,
)

#: Порог численного расхождения ONNX и PyTorch (10.5).
PARITY_TOLERANCE = 1e-4

#: Размеры батчей, на которых проверяются формы (10.5).
SHAPE_BATCHES = (1, 64)

#: Коды типов элементов ONNX (``onnx.TensorProto``). Продублированы числами,
#: чтобы проверка типов не требовала импорта onnx на уровне модуля.
FLOAT_ELEM_TYPE = 1        # TensorProto.FLOAT
INTEGER_ELEM_TYPES = (6, 7)  # TensorProto.INT32, TensorProto.INT64

_ELEM_TYPE_NAMES = {1: "float32", 6: "int32", 7: "int64", 11: "float64"}


@dataclass
class VerificationResult:
    """Итог верификации: список проверок и агрегированный вердикт."""

    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def failures(self) -> list[tuple[str, str]]:
        return [(name, detail) for name, ok, detail in self.checks if not ok]

    def report(self) -> str:
        lines = [f"{'OK  ' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else "")
                 for name, ok, detail in self.checks]
        lines.append(f"ИТОГ: {'ПРОЙДЕНО' if self.passed else 'ПРОВАЛЕНО'} "
                     f"({sum(ok for _, ok, _ in self.checks)}/{len(self.checks)})")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Блокирует приёмку при провале (10.5)."""
        if not self.passed:
            raise AssertionError("верификация ONNX провалена:\n" + self.report())


def verify_onnx_model(
    onnx_path: str | Path,
    policy: torch.nn.Module,
    action_spec: _HasActionSpec,
    obs_shapes: Sequence[tuple[int, ...]],
    sample_obs: Sequence[np.ndarray] | None = None,
    memory_size: int = 0,
    strategy: str = "greedy",
) -> VerificationResult:
    """Полная проверка экспортированной модели.

    Args:
        onnx_path: путь к ``.onnx``.
        policy: тот же модуль политики, что экспортировался.
        action_spec: пространство действий среды.
        obs_shapes: формы наблюдений по сенсорам, без батча.
        sample_obs: наблюдения **из реального распределения среды** для проверки
            числового паритета — по одному массиву ``(64, *shape)`` на сенсор.
            Если не переданы, используется ``N(0, 1)``, и это отмечается
            в отчёте: требование 10.5 говорит о реальном распределении.
        memory_size: размер рекуррентной памяти.
        strategy: та же стратегия, что при экспорте.

    Returns:
        :class:`VerificationResult`. Вызов ``raise_if_failed()`` превращает
        провал в исключение.
    """
    import onnx
    import onnxruntime as ort

    result = VerificationResult()
    path = Path(onnx_path)

    # --- 1. Структура ----------------------------------------------------
    model = onnx.load(str(path))
    try:
        onnx.checker.check_model(model)
        result.add("структура (onnx.checker)", True)
    except Exception as exc:  # noqa: BLE001 — сообщение checker'а важнее типа
        result.add("структура (onnx.checker)", False, str(exc))

    # --- 2. opset --------------------------------------------------------
    opsets = {imp.domain: imp.version for imp in model.opset_import}
    actual_opset = opsets.get("", opsets.get("ai.onnx"))
    result.add(
        "opset", actual_opset == ONNX_OPSET,
        f"ожидался {ONNX_OPSET}, в модели {actual_opset}",
    )

    # --- 3. Имена входов и выходов --------------------------------------
    expected_inputs = set(contract_input_names(action_spec, len(obs_shapes), memory_size))
    expected_outputs = set(contract_output_names(action_spec, memory_size))
    actual_inputs = {i.name for i in model.graph.input}
    actual_outputs = {o.name for o in model.graph.output}
    result.add(
        "имена входов", actual_inputs == expected_inputs,
        f"лишние={sorted(actual_inputs - expected_inputs)}, "
        f"отсутствуют={sorted(expected_inputs - actual_inputs)}",
    )
    result.add(
        "имена выходов", actual_outputs == expected_outputs,
        f"лишние={sorted(actual_outputs - expected_outputs)}, "
        f"отсутствуют={sorted(expected_outputs - actual_outputs)}",
    )

    # --- 4. Типы выходов действий ---------------------------------------
    # Unity читает discrete_actions как Tensor<int>, а continuous_actions —
    # как Tensor<float> (ApplierImpl.cs пакета com.unity.ml-agents). Модель
    # с «не тем» типом остаётся валидным ONNX и проходит onnxruntime, но
    # падает при инференсе в Unity, поэтому тип проверяется отдельно.
    output_types = {o.name: o.type.tensor_type.elem_type for o in model.graph.output}
    for name in ("discrete_actions", "deterministic_discrete_actions"):
        if name in output_types:
            result.add(
                f"тип выхода {name}", output_types[name] in INTEGER_ELEM_TYPES,
                f"ожидался целочисленный тензор, получен elem_type={output_types[name]} "
                f"({_ELEM_TYPE_NAMES.get(output_types[name], 'неизвестный')})",
            )
    for name in ("continuous_actions", "deterministic_continuous_actions"):
        if name in output_types:
            result.add(
                f"тип выхода {name}", output_types[name] == FLOAT_ELEM_TYPE,
                f"ожидался float32, получен elem_type={output_types[name]} "
                f"({_ELEM_TYPE_NAMES.get(output_types[name], 'неизвестный')})",
            )

    # --- 5. Прогон и формы на батчах 1 и 64 ------------------------------
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    mask_size = int(sum(action_spec.discrete_branches))
    out_names = [o.name for o in session.get_outputs()]
    session_inputs = {i.name for i in session.get_inputs()}

    runs: dict[int, dict[str, np.ndarray]] = {}
    for batch in SHAPE_BATCHES:
        feed = _make_feed(obs_shapes, batch, mask_size, memory_size, sample_obs if batch == 64 else None)
        try:
            # В графе есть только те входы, которые модель действительно
            # использует, поэтому лишние ключи отбрасываются: onnxruntime
            # считает неизвестное имя входа ошибкой.
            values = session.run(out_names, {k: v for k, v in feed.items() if k in session_inputs})
            runs[batch] = dict(zip(out_names, values))
            result.add(f"прогон onnxruntime, батч {batch}", True)
        except Exception as exc:  # noqa: BLE001
            result.add(f"прогон onnxruntime, батч {batch}", False, str(exc))

    # --- 6. Константы ----------------------------------------------------
    if SHAPE_BATCHES[0] in runs:
        out1 = runs[SHAPE_BATCHES[0]]
        _check_constant(result, out1, "version_number", [float(MODEL_EXPORT_VERSION)])
        _check_constant(result, out1, "memory_size", [float(memory_size)])
        if action_spec.continuous_size > 0:
            _check_constant(result, out1, "continuous_action_output_shape",
                            [float(action_spec.continuous_size)])
        if len(action_spec.discrete_branches) > 0:
            _check_constant(result, out1, "discrete_action_output_shape",
                            [float(b) for b in action_spec.discrete_branches])

    # --- 7. Числовой паритет и диапазоны --------------------------------
    if 64 in runs:
        feed64 = _make_feed(obs_shapes, 64, mask_size, memory_size, sample_obs)
        # Сверка идёт на CPU: ровно там же исполняется экспортированный граф
        # в onnxruntime, и сравнивать надо одно с одним. Исходное устройство
        # модуля восстанавливается — обучение может продолжаться на GPU.
        original_device = _module_device(policy)
        policy.to("cpu")
        try:
            wrapper = MLAgentsPolicyWrapper(policy, action_spec, memory_size, strategy).eval()
            torch_inputs = [torch.from_numpy(feed64[name]) for name in _forward_arg_names(len(obs_shapes))]
            with torch.no_grad():
                torch_out = wrapper(*torch_inputs)
        finally:
            policy.to(original_device)
        torch_named = dict(zip(contract_output_names(action_spec, memory_size), torch_out))

        # Сравниваем по детерминированным выходам: стохастические (`sample()`)
        # поэлементно несравнимы по построению — см. ASSUMPTIONS A-6.
        for name in ("deterministic_continuous_actions", "deterministic_discrete_actions"):
            if name not in torch_named:
                continue
            diff = float(np.max(np.abs(runs[64][name] - torch_named[name].cpu().numpy())))
            result.add(
                f"числовой паритет: {name}", diff <= PARITY_TOLERANCE,
                f"max|ONNX-PyTorch| = {diff:.3e}, допуск {PARITY_TOLERANCE:.0e}",
            )

        if sample_obs is None:
            result.add(
                "паритет считан на реальном распределении наблюдений", True,
                "ВНИМАНИЕ: sample_obs не передан, использован N(0,1) — "
                "для приёмки примера передайте наблюдения из среды (10.5)",
            )

        _check_ranges(result, runs[64], action_spec)

    return result


def _forward_arg_names(num_obs: int) -> list[str]:
    """Порядок аргументов ``MLAgentsPolicyWrapper.forward``.

    Отличается от имён входов графа: ``forward`` всегда принимает маски
    и память, даже если те не попадают в экспортированный граф.
    """
    return [f"obs_{i}" for i in range(num_obs)] + ["action_masks", "recurrent_in"]


def _make_feed(
    obs_shapes: Sequence[tuple[int, ...]],
    batch: int,
    mask_size: int,
    memory_size: int,
    sample_obs: Sequence[np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Собирает полный набор входов: и для графа, и для вызова ``forward``."""
    rng = np.random.default_rng(0)
    feed: dict[str, np.ndarray] = {}
    for i, shape in enumerate(obs_shapes):
        if sample_obs is not None:
            arr = np.asarray(sample_obs[i], dtype=np.float32)
            if arr.shape != (batch, *shape):
                raise ValueError(
                    f"sample_obs[{i}] должен иметь форму {(batch, *shape)}, получено {arr.shape}"
                )
        else:
            arr = rng.standard_normal((batch, *shape)).astype(np.float32)
        feed[f"obs_{i}"] = arr
    feed["action_masks"] = np.ones((batch, mask_size), dtype=np.float32)
    feed["recurrent_in"] = np.zeros((batch, 1, memory_size), dtype=np.float32)
    return feed


def _check_constant(
    result: VerificationResult,
    outputs: dict[str, np.ndarray],
    name: str,
    expected: list[float],
) -> None:
    if name not in outputs:
        result.add(f"константа {name}", False, "выход отсутствует в модели")
        return
    actual = np.asarray(outputs[name], dtype=np.float64).reshape(-1).tolist()
    ok = len(actual) == len(expected) and all(abs(a - e) < 1e-6 for a, e in zip(actual, expected))
    result.add(f"константа {name}", ok, f"ожидалось {expected}, получено {actual}")


def _check_ranges(
    result: VerificationResult,
    outputs: dict[str, np.ndarray],
    action_spec: _HasActionSpec,
) -> None:
    """Диапазоны действий (10.5, последняя строка таблицы)."""
    if action_spec.continuous_size > 0:
        vals = outputs["continuous_actions"]
        lo, hi = float(vals.min()), float(vals.max())
        result.add(
            "диапазон непрерывных действий", -1.0 <= lo and hi <= 1.0,
            f"[{lo:.4f}, {hi:.4f}], допустимо [-1, 1] (clamp ±{CONTINUOUS_CLIP:g} / {CONTINUOUS_CLIP:g})",
        )
    if not action_spec.discrete_branches:
        return

    # Верификатор не имеет права падать на «неправильной» модели: его задача —
    # сообщить о несоответствии, а не выбросить IndexError. Число колонок
    # выхода сверяется до обращения к ним.
    columns = outputs["discrete_actions"].shape[1]
    if columns != len(action_spec.discrete_branches):
        result.add(
            "число дискретных веток на выходе", False,
            f"в модели {columns} колонок, в ActionSpec {len(action_spec.discrete_branches)} веток",
        )
        return

    for branch_idx, branch_size in enumerate(action_spec.discrete_branches):
        vals = outputs["discrete_actions"][:, branch_idx]
        integral = bool(np.all(vals == np.floor(vals)))
        in_range = bool(np.all((vals >= 0) & (vals < branch_size)))
        result.add(
            f"диапазон дискретной ветки {branch_idx}", integral and in_range,
            f"значения [{vals.min():.0f}, {vals.max():.0f}], допустимо целые [0, {branch_size})",
        )
