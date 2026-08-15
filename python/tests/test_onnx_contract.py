"""Контракт экспорта ONNX (раздел 10 инструкции) — критический тест каркаса.

Проверяется то, что ломает весь замысел проекта, если сломано: имена входов
и выходов, значения констант, семантика действий и совпадение ONNX с PyTorch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from labrl.export.onnx_export import (
    MODEL_EXPORT_VERSION,
    ONNX_OPSET,
    ActionSpecLite,
    MLAgentsPolicyWrapper,
    contract_input_names,
    contract_output_names,
    export_policy_to_onnx,
)
from labrl.export.onnx_verify import verify_onnx_model

DISCRETE_SPEC = ActionSpecLite(discrete_branches=(4,))
TWO_BRANCH_SPEC = ActionSpecLite(discrete_branches=(3, 2))
CONTINUOUS_SPEC = ActionSpecLite(continuous_size=2)


def tiny_net(in_dim: int, out_dim: int) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, out_dim))


# --- имена контракта ----------------------------------------------------


def test_input_names_follow_contract():
    """Маски входят в граф только у дискретной политики, память — только у рекуррентной.

    Это не упрощение: torch.onnx.export выбрасывает неиспользуемые входы,
    и Unity ждёт ровно такой граф (SentisModelParamLoader.CheckInputTensorPresence).
    """
    assert contract_input_names(DISCRETE_SPEC, 1) == ["obs_0", "action_masks"]
    assert contract_input_names(DISCRETE_SPEC, 3) == ["obs_0", "obs_1", "obs_2", "action_masks"]
    assert contract_input_names(CONTINUOUS_SPEC, 1) == ["obs_0"]
    assert contract_input_names(DISCRETE_SPEC, 1, memory_size=8) == ["obs_0", "action_masks", "recurrent_in"]


def test_output_names_and_order_for_discrete():
    assert contract_output_names(DISCRETE_SPEC) == [
        "version_number", "memory_size",
        "discrete_actions", "discrete_action_output_shape", "deterministic_discrete_actions",
    ]


def test_output_names_and_order_for_continuous():
    assert contract_output_names(CONTINUOUS_SPEC) == [
        "version_number", "memory_size",
        "continuous_actions", "continuous_action_output_shape", "deterministic_continuous_actions",
    ]


# --- поведение обёртки --------------------------------------------------


def test_wrapper_constants_have_contract_values():
    w = MLAgentsPolicyWrapper(tiny_net(25, 4), DISCRETE_SPEC)
    assert w.version_number.tolist() == [float(MODEL_EXPORT_VERSION)]
    assert w.memory_size_vector.tolist() == [0.0]
    assert w.discrete_act_size_vector.tolist() == [[4.0]]


def test_wrapper_returns_action_indices_not_logits():
    """Требование 10.4: для дискретных действий выдаются индексы, не логиты."""
    w = MLAgentsPolicyWrapper(tiny_net(6, 4), DISCRETE_SPEC).eval()
    out = dict(zip(contract_output_names(DISCRETE_SPEC), w(torch.randn(8, 6), torch.ones(8, 4), torch.zeros(8, 1, 0))))

    actions = out["discrete_actions"]
    assert actions.shape == (8, 1)
    assert torch.all((actions >= 0) & (actions < 4))
    assert torch.all(actions == actions.floor())


def test_wrapper_respects_action_masks():
    """Замаскированное действие не должно выбираться, даже если у него max Q."""
    class Constant(nn.Module):
        def forward(self, obs):
            # Максимум всегда у действия 0 — маска обязана его перебить.
            return torch.tensor([[10.0, 1.0, 2.0, 3.0]]).repeat(obs.shape[0], 1)

    w = MLAgentsPolicyWrapper(Constant(), DISCRETE_SPEC).eval()
    masks = torch.tensor([[0.0, 1.0, 1.0, 1.0]]).repeat(4, 1)
    out = dict(zip(contract_output_names(DISCRETE_SPEC), w(torch.randn(4, 6), masks, torch.zeros(4, 1, 0))))

    assert torch.all(out["deterministic_discrete_actions"] == 3)


def test_wrapper_clips_continuous_actions_to_unit_range():
    """Контракт §4.2: clamp(±3) / 3 — Unity получает действия в [-1, 1]."""
    class Huge(nn.Module):
        def forward(self, obs):
            return torch.tensor([[100.0, -100.0]]).repeat(obs.shape[0], 1)

    w = MLAgentsPolicyWrapper(Huge(), CONTINUOUS_SPEC).eval()
    out = dict(zip(contract_output_names(CONTINUOUS_SPEC), w(torch.randn(4, 8), torch.ones(4, 0), torch.zeros(4, 1, 0))))

    assert torch.allclose(out["continuous_actions"], torch.tensor([[1.0, -1.0]]).repeat(4, 1))


def test_wrapper_handles_multiple_discrete_branches():
    w = MLAgentsPolicyWrapper(tiny_net(6, 5), TWO_BRANCH_SPEC).eval()
    out = dict(zip(contract_output_names(TWO_BRANCH_SPEC), w(torch.randn(4, 6), torch.ones(4, 5), torch.zeros(4, 1, 0))))

    actions = out["discrete_actions"]
    assert actions.shape == (4, 2)
    assert torch.all((actions[:, 0] >= 0) & (actions[:, 0] < 3))
    assert torch.all((actions[:, 1] >= 0) & (actions[:, 1] < 2))


def test_wrapper_rejects_recurrent_and_hybrid_spaces():
    with pytest.raises(NotImplementedError, match="рекуррентные"):
        MLAgentsPolicyWrapper(tiny_net(4, 4), DISCRETE_SPEC, memory_size=8)
    with pytest.raises(NotImplementedError, match="гибридное"):
        MLAgentsPolicyWrapper(tiny_net(4, 6), ActionSpecLite(continuous_size=2, discrete_branches=(4,)))


# --- сквозной экспорт и верификация -------------------------------------


def test_discrete_export_passes_full_verification(tmp_path):
    policy = tiny_net(25, 4)
    obs_shapes = [(25,)]
    path = export_policy_to_onnx(policy, DISCRETE_SPEC, obs_shapes, tmp_path / "d.onnx")

    rng = np.random.default_rng(0)
    sample = [rng.standard_normal((64, 25)).astype(np.float32)]
    result = verify_onnx_model(path, policy, DISCRETE_SPEC, obs_shapes, sample_obs=sample)

    assert result.passed, result.report()


def test_continuous_export_passes_full_verification(tmp_path):
    policy = tiny_net(8, 2)
    obs_shapes = [(8,)]
    path = export_policy_to_onnx(policy, CONTINUOUS_SPEC, obs_shapes, tmp_path / "c.onnx")

    rng = np.random.default_rng(1)
    sample = [rng.standard_normal((64, 8)).astype(np.float32)]
    result = verify_onnx_model(path, policy, CONTINUOUS_SPEC, obs_shapes, sample_obs=sample)

    assert result.passed, result.report()


def test_discrete_action_outputs_are_integer_tensors(tmp_path):
    """Unity читает discrete_actions как Tensor<int>.

    Модель с float-выходом остаётся валидным ONNX и проходит onnxruntime,
    но падает при инференсе в Unity (`ApplierImpl.cs`,
    `DiscreteActionOutputApplier`). Поэтому тип проверяется здесь.
    """
    import onnx

    path = export_policy_to_onnx(tiny_net(25, 4), DISCRETE_SPEC, [(25,)], tmp_path / "int.onnx")
    types = {o.name: o.type.tensor_type.elem_type for o in onnx.load(str(path)).graph.output}

    integer_types = (onnx.TensorProto.INT32, onnx.TensorProto.INT64)
    assert types["discrete_actions"] in integer_types
    assert types["deterministic_discrete_actions"] in integer_types


def test_continuous_action_outputs_are_float_tensors(tmp_path):
    import onnx

    path = export_policy_to_onnx(tiny_net(8, 2), CONTINUOUS_SPEC, [(8,)], tmp_path / "float.onnx")
    types = {o.name: o.type.tensor_type.elem_type for o in onnx.load(str(path)).graph.output}

    assert types["continuous_actions"] == onnx.TensorProto.FLOAT
    assert types["deterministic_continuous_actions"] == onnx.TensorProto.FLOAT


def test_exported_graph_uses_required_opset(tmp_path):
    import onnx

    path = export_policy_to_onnx(tiny_net(25, 4), DISCRETE_SPEC, [(25,)], tmp_path / "o.onnx")
    model = onnx.load(str(path))
    opsets = {imp.domain: imp.version for imp in model.opset_import}

    assert opsets.get("", opsets.get("ai.onnx")) == ONNX_OPSET


def test_exported_graph_accepts_dynamic_batch(tmp_path):
    """Unity подаёт батч переменного размера — фиксированный батч сломает инференс."""
    import onnxruntime as ort

    path = export_policy_to_onnx(tiny_net(25, 4), DISCRETE_SPEC, [(25,)], tmp_path / "b.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for batch in (1, 7, 64):
        outputs = session.run(
            ["discrete_actions"],
            {
                "obs_0": np.zeros((batch, 25), dtype=np.float32),
                "action_masks": np.ones((batch, 4), dtype=np.float32),
            },
        )
        assert outputs[0].shape == (batch, 1)


def test_verification_detects_wrong_action_spec(tmp_path):
    """Верификация обязана падать при расхождении спецификации и модели.

    Ветки (2, 2) дают ту же суммарную ширину масок, что и (4,), поэтому граф
    прогоняется без ошибок — и расхождение ловится именно проверкой константы
    `discrete_action_output_shape`, а не побочным падением прогона.
    """
    policy = tiny_net(25, 4)
    path = export_policy_to_onnx(policy, DISCRETE_SPEC, [(25,)], tmp_path / "w.onnx")

    wrong = ActionSpecLite(discrete_branches=(2, 2))
    result = verify_onnx_model(path, policy, wrong, [(25,)])

    assert not result.passed
    assert any("discrete_action_output_shape" in name for name, _ in result.failures), result.report()


def test_failed_verification_blocks_acceptance(tmp_path):
    """Провал верификации обязан быть исключением, а не строчкой в логе (10.5)."""
    policy = tiny_net(25, 4)
    path = export_policy_to_onnx(policy, DISCRETE_SPEC, [(25,)], tmp_path / "f.onnx")
    result = verify_onnx_model(path, policy, ActionSpecLite(discrete_branches=(2, 2)), [(25,)])

    with pytest.raises(AssertionError, match="верификация ONNX провалена"):
        result.raise_if_failed()
