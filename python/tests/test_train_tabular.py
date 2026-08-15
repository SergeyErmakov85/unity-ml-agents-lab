"""Сбор опыта в цикле обучения: что попадает в батч, а что нет.

Проверяется на поддельной среде — ошибка здесь не проявляется как падение,
а тихо портит обучающую выборку.
"""

from __future__ import annotations

import numpy as np
import pytest

from labrl.envs.vec_unity_env import StepResult
from labrl.train.tabular import _build_batch, epsilon_greedy_entropy, states_from_obs


def one_hot(indices: list[int], size: int = 5) -> np.ndarray:
    out = np.zeros((len(indices), size), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def make_result(**kwargs) -> StepResult:
    n = len(kwargs["reward"])
    defaults = dict(
        obs=[one_hot([0] * n)],
        final_obs=[one_hot([0] * n)],
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        active=np.ones(n, dtype=bool),
        info={},
    )
    defaults.update(kwargs)
    return StepResult(**defaults)


def test_states_from_obs_decodes_one_hot():
    obs = [one_hot([3, 0, 4])]
    assert states_from_obs(obs).tolist() == [3, 0, 4]


def test_batch_uses_final_obs_for_finished_episodes():
    """Для завершённого эпизода следующее состояние берётся из final_obs.

    В `obs` там уже может лежать начало нового эпизода — использовать его
    значило бы считать цель обновления по состоянию из другой траектории.
    """
    result = make_result(
        reward=np.array([1.0, -0.04]),
        obs=[one_hot([0, 2])],        # слот 0 уже начал новый эпизод в клетке 0
        final_obs=[one_hot([4, 0])],  # но закончил он в клетке 4
        terminated=np.array([True, False]),
        active=np.array([True, True]),
    )
    batch = _build_batch(np.array([3, 1]), np.array([0, 2]), result)

    assert batch.next_state.tolist() == [4, 2]
    assert batch.terminated.tolist() == [True, False]


def test_inactive_slot_without_result_is_excluded():
    """Слот, который не получил ни решения, ни терминала, перехода не даёт."""
    result = make_result(
        reward=np.array([0.0, -0.04]),
        active=np.array([False, True]),
    )
    batch = _build_batch(np.array([1, 2]), np.array([0, 1]), result)

    assert batch.state.tolist() == [2]
    assert batch.action.tolist() == [1]


def test_truncated_slot_is_included_and_marked():
    result = make_result(
        reward=np.array([-0.04]),
        final_obs=[one_hot([2])],
        truncated=np.array([True]),
        active=np.array([False]),
    )
    batch = _build_batch(np.array([1]), np.array([3]), result)

    assert batch.state.tolist() == [1]
    assert batch.terminated.tolist() == [False]
    assert batch.truncated.tolist() == [True]
    assert batch.next_state.tolist() == [2]


def test_empty_batch_when_nothing_returned():
    result = make_result(reward=np.array([0.0]), active=np.array([False]))
    batch = _build_batch(np.array([1]), np.array([0]), result)
    assert batch.state.size == 0


def test_entropy_spans_from_zero_to_log_a():
    """ε=0 — чистая эксплуатация, ε=1 — равномерная политика."""
    assert epsilon_greedy_entropy(0.0, 4) == pytest.approx(0.0)
    assert epsilon_greedy_entropy(1.0, 4) == pytest.approx(np.log(4))
    mid = epsilon_greedy_entropy(0.5, 4)
    assert 0.0 < mid < np.log(4)


def test_entropy_is_monotonic_in_epsilon():
    values = [epsilon_greedy_entropy(e, 4) for e in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)
