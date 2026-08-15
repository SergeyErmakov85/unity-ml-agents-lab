"""Дискретизация непрерывного наблюдения и её присутствие в графе ONNX (`E02`).

Здесь проверяется то, что рушит пример незаметно: сетка, применённая
в Python и в Unity по-разному. Требование 10.7 запрещает держать
преобразование входа вне графа, и тесты ниже — механическая проверка того,
что запрет соблюдён.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from labrl.algos.tabular.q_learning import QLearning
from labrl.envs.state_encoders import BoxDiscretizer, OneHotStateEncoder
from labrl.export.onnx_export import ActionSpecLite, export_policy_to_onnx
from labrl.export.onnx_verify import verify_onnx_model
from labrl.nets.discretized import DiscretizedQTable

GRID = BoxDiscretizer.uniform(lows=[-1.0, -2.0], highs=[1.0, 2.0], bins=[2, 4])


# --- геометрия сетки ----------------------------------------------------


def test_uniform_builds_inner_boundaries_only():
    """n ячеек задаются n−1 внутренними границами: края разомкнуты."""
    assert GRID.boundaries[0] == (0.0,)
    assert GRID.boundaries[1] == pytest.approx((-1.0, 0.0, 1.0))
    assert GRID.bins_per_dim == (2, 4)
    assert GRID.num_states == 8


def test_strides_are_row_major():
    """Последнее измерение меняется быстрее — как в C-массивах."""
    assert GRID.strides == (4, 1)


def test_index_matches_manual_computation():
    obs = np.array([[-0.5, -1.5], [0.5, 1.5], [0.0, 0.0]])
    # Ячейка = число превышенных границ. Границы измерений: (0.0) и (-1, 0, 1).
    expected = np.array([0 * 4 + 0, 1 * 4 + 3, 0 * 4 + 1])
    np.testing.assert_array_equal(GRID.index(obs), expected)


def test_values_outside_range_fall_into_edge_cells():
    """Сетку не требуется строить по экстремумам: хвосты сливаются в края."""
    obs = np.array([[-100.0, -100.0], [100.0, 100.0]])
    np.testing.assert_array_equal(GRID.index(obs), np.array([0, GRID.num_states - 1]))


def test_boundary_value_goes_to_upper_cell_only_when_strictly_greater():
    """Правило сравнения строгое: значение ровно на границе остаётся снизу."""
    np.testing.assert_array_equal(GRID.index(np.array([[0.0, -1.0]])), np.array([0 * 4 + 0]))
    np.testing.assert_array_equal(GRID.index(np.array([[1e-6, -1.0]])), np.array([1 * 4 + 0]))


def test_degenerate_grid_ignores_dimension():
    """Одна ячейка на ось означает «это измерение не влияет на состояние»."""
    grid = BoxDiscretizer.uniform(lows=[0.0, 0.0], highs=[1.0, 1.0], bins=[1, 3])
    assert grid.num_states == 3
    assert grid.index(np.array([[-99.0, 0.5]]))[0] == grid.index(np.array([[99.0, 0.5]]))[0]


def test_uniform_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="одной длины"):
        BoxDiscretizer.uniform(lows=[0.0], highs=[1.0, 2.0], bins=[2, 2])


# --- модуль для графа ---------------------------------------------------


def random_table(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(GRID.num_states, 3)).astype(np.float32)


def test_module_reproduces_table_lookup():
    """Граф обязан выдавать ту же строку таблицы, что и numpy-дискретизатор."""
    rng = np.random.default_rng(0)
    q = random_table(rng)
    module = DiscretizedQTable.from_table(GRID, q)

    obs = rng.uniform(-3.0, 3.0, size=(64, 2)).astype(np.float32)
    out = module(torch.from_numpy(obs)).detach().numpy()

    np.testing.assert_allclose(out, q[GRID.index(obs)], rtol=0, atol=1e-6)


def test_module_rejects_table_of_wrong_size():
    with pytest.raises(ValueError, match="состояний"):
        DiscretizedQTable.from_table(GRID, np.zeros((GRID.num_states + 1, 2)))


def test_to_table_round_trip():
    rng = np.random.default_rng(1)
    q = random_table(rng)
    np.testing.assert_allclose(DiscretizedQTable.from_table(GRID, q).to_table(), q)


# --- связка с алгоритмом ------------------------------------------------


def test_q_learning_uses_encoder_for_states_and_export():
    algo = QLearning(num_states=GRID.num_states, num_actions=3, encoder=GRID)
    algo.q[:] = np.random.default_rng(2).normal(size=algo.q.shape)

    module = algo.policy_module()
    assert isinstance(module, DiscretizedQTable)

    obs = np.array([[0.5, 1.5], [-0.5, -1.5]], dtype=np.float32)
    greedy = algo.greedy_action(GRID.index(obs))
    graph = module(torch.from_numpy(obs)).argmax(dim=1).numpy()
    np.testing.assert_array_equal(graph, greedy)


def test_encoder_size_must_match_table():
    with pytest.raises(ValueError, match="кодировщик"):
        QLearning(num_states=GRID.num_states + 1, num_actions=3, encoder=GRID)


def test_default_encoder_is_one_hot():
    algo = QLearning(num_states=5, num_actions=2)
    assert isinstance(algo.encoder, OneHotStateEncoder)
    np.testing.assert_array_equal(algo.encoder.index(np.eye(5)), np.arange(5))


# --- экспорт в ONNX -----------------------------------------------------


def test_discretized_policy_passes_onnx_contract(tmp_path):
    """Экспорт сетки вместе с таблицей проходит все проверки раздела 10.5.

    Это и есть доказательство, что дискретизация уехала в граф: числовой
    паритет считается по **сырым** наблюдениям, а не по индексам состояний.
    """
    rng = np.random.default_rng(3)
    q = random_table(rng)
    module = DiscretizedQTable.from_table(GRID, q)
    spec = ActionSpecLite(discrete_branches=(3,))

    path = export_policy_to_onnx(module, spec, [(2,)], tmp_path / "policy.onnx")
    sample = [rng.uniform(-3.0, 3.0, size=(64, 2)).astype(np.float32)]
    result = verify_onnx_model(path, module, spec, [(2,)], sample_obs=sample)

    assert result.passed, result.report()
