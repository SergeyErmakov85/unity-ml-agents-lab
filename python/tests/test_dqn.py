"""DQN и буфер воспроизведения.

Проверяется то, что ломается тихо: различение terminated/truncated в цели
обновления, роль целевой сети, корректность кольцевого буфера и сходимость
на игрушечной задаче с известным ответом.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from labrl.algos.dqn import DQN, DQNConfig
from labrl.buffers.replay import Batch, ReplayBuffer
from labrl.envs.vec_unity_env import StepResult
from labrl.nets.mlp import MLPQNetwork
from labrl.train.dqn import _store_transitions, epsilon_greedy_entropy


def make_algo(obs_dim: int = 4, num_actions: int = 3, **cfg_kwargs) -> DQN:
    torch.manual_seed(0)
    q = MLPQNetwork(obs_dim, num_actions, hidden_sizes=(16,))
    target = MLPQNetwork(obs_dim, num_actions, hidden_sizes=(16,))
    return DQN(q, target, DQNConfig(**cfg_kwargs))


# --- буфер ---------------------------------------------------------------


def test_buffer_grows_and_caps_at_capacity():
    buf = ReplayBuffer(capacity=5, obs_dim=2)
    for i in range(7):
        buf.add_batch(
            obs=np.full((1, 2), i, dtype=np.float32),
            action=np.array([i % 3]),
            reward=np.array([float(i)], dtype=np.float32),
            next_obs=np.full((1, 2), i + 1, dtype=np.float32),
            terminated=np.array([False]),
            truncated=np.array([False]),
        )
    assert len(buf) == 5
    assert buf.is_full


def test_buffer_overwrites_oldest_entries():
    """Кольцевой буфер обязан затирать старое, а не молча терять новое."""
    buf = ReplayBuffer(capacity=3, obs_dim=1)
    for i in range(5):
        buf.add_batch(
            obs=np.array([[float(i)]], dtype=np.float32),
            action=np.array([0]),
            reward=np.array([float(i)], dtype=np.float32),
            next_obs=np.array([[float(i)]], dtype=np.float32),
            terminated=np.array([False]),
            truncated=np.array([False]),
        )
    sample = buf.sample(200)
    seen = set(np.unique(sample.reward).tolist())
    assert seen <= {2.0, 3.0, 4.0}, f"в буфере остались затёртые переходы: {seen}"


def test_buffer_write_wraps_around_the_end():
    """Пачка, пересекающая конец кольца, должна лечь целиком."""
    buf = ReplayBuffer(capacity=4, obs_dim=1)
    buf.add_batch(
        obs=np.arange(3, dtype=np.float32).reshape(3, 1),
        action=np.zeros(3, dtype=np.int64),
        reward=np.arange(3, dtype=np.float32),
        next_obs=np.arange(3, dtype=np.float32).reshape(3, 1),
        terminated=np.zeros(3, dtype=bool),
        truncated=np.zeros(3, dtype=bool),
    )
    buf.add_batch(
        obs=np.arange(3, 6, dtype=np.float32).reshape(3, 1),
        action=np.zeros(3, dtype=np.int64),
        reward=np.arange(3, 6, dtype=np.float32),
        next_obs=np.arange(3, 6, dtype=np.float32).reshape(3, 1),
        terminated=np.zeros(3, dtype=bool),
        truncated=np.zeros(3, dtype=bool),
    )
    assert len(buf) == 4
    seen = set(np.unique(buf.sample(300).reward).tolist())
    assert seen == {2.0, 3.0, 4.0, 5.0}


def test_buffer_keeps_terminated_and_truncated_apart():
    buf = ReplayBuffer(capacity=2, obs_dim=1)
    buf.add_batch(
        obs=np.zeros((2, 1), dtype=np.float32),
        action=np.zeros(2, dtype=np.int64),
        reward=np.zeros(2, dtype=np.float32),
        next_obs=np.zeros((2, 1), dtype=np.float32),
        terminated=np.array([True, False]),
        truncated=np.array([False, True]),
    )
    sample = buf.sample(200)
    assert set(zip(sample.terminated.tolist(), sample.truncated.tolist())) == {(True, False), (False, True)}


def test_empty_buffer_sample_is_an_error():
    with pytest.raises(RuntimeError, match="буфер пуст"):
        ReplayBuffer(4, 2).sample(1)


# --- цель обновления -----------------------------------------------------


def one_step_batch(terminated: bool, truncated: bool) -> Batch:
    return Batch(
        obs=np.zeros((1, 4), dtype=np.float32),
        action=np.array([0]),
        reward=np.array([1.0], dtype=np.float32),
        next_obs=np.ones((1, 4), dtype=np.float32),
        terminated=np.array([terminated]),
        truncated=np.array([truncated]),
    )


def test_terminated_zeroes_the_future_but_truncated_does_not():
    """Ключевое различие метода: обрыв по времени не обнуляет будущее."""
    algo = make_algo(gamma=0.9, double_dqn=False)

    with torch.no_grad():
        next_q = algo.target_net(torch.ones(1, 4)).max().item()

    def target_of(batch: Batch) -> float:
        obs = torch.as_tensor(batch.next_obs)
        with torch.no_grad():
            nq = algo.target_net(obs).max(dim=1).values
        term = torch.as_tensor(batch.terminated)
        return float((torch.as_tensor(batch.reward) + 0.9 * nq * (~term).float()).item())

    assert target_of(one_step_batch(terminated=True, truncated=False)) == pytest.approx(1.0)
    assert target_of(one_step_batch(terminated=False, truncated=True)) == pytest.approx(1.0 + 0.9 * next_q, rel=1e-5)


def test_update_returns_metrics_and_advances_counter():
    algo = make_algo()
    metrics = algo.update(one_step_batch(False, False))

    assert algo.updates == 1
    for key in ("loss", "td_error_abs", "q_mean", "q_max", "grad_norm"):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_target_network_is_frozen_between_syncs():
    """Целевая сеть не должна двигаться от градиентных шагов — в этом её смысл."""
    algo = make_algo(target_update_interval=1000)
    before = [p.clone() for p in algo.target_net.parameters()]

    for _ in range(5):
        algo.update(one_step_batch(False, False))

    after = list(algo.target_net.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_target_network_syncs_on_schedule():
    algo = make_algo(target_update_interval=3)
    for _ in range(3):
        algo.update(one_step_batch(False, False))

    for online, target in zip(algo.q_net.parameters(), algo.target_net.parameters()):
        assert torch.equal(online, target)


def test_online_network_actually_changes():
    algo = make_algo()
    before = [p.clone() for p in algo.q_net.parameters()]
    algo.update(one_step_batch(False, False))
    after = list(algo.q_net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_double_dqn_and_plain_dqn_both_run():
    for double in (True, False):
        algo = make_algo(double_dqn=double)
        metrics = algo.update(one_step_batch(False, False))
        assert np.isfinite(metrics["loss"])


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="gamma"):
        DQNConfig(gamma=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        DQNConfig(batch_size=0)
    with pytest.raises(ValueError, match="target_update_interval"):
        DQNConfig(target_update_interval=0)


# --- политика ------------------------------------------------------------


def test_epsilon_zero_is_purely_greedy():
    algo = make_algo(obs_dim=2, num_actions=3)

    class Constant(nn.Module):
        def forward(self, obs):
            return torch.tensor([[0.0, 5.0, 0.0]]).repeat(obs.shape[0], 1)

    algo.q_net = Constant()
    rng = np.random.default_rng(0)
    actions = algo.act(np.zeros((32, 2), dtype=np.float32), epsilon=0.0, rng=rng)
    assert np.all(actions == 1)


def test_epsilon_one_explores_all_actions():
    algo = make_algo(obs_dim=2, num_actions=3)
    rng = np.random.default_rng(0)
    actions = algo.act(np.zeros((300, 2), dtype=np.float32), epsilon=1.0, rng=rng)
    assert set(np.unique(actions).tolist()) == {0, 1, 2}


def test_num_actions_is_inferred_from_the_network():
    assert make_algo(obs_dim=6, num_actions=5).num_actions == 5


# --- сходимость на игрушечной задаче ------------------------------------


def test_dqn_learns_a_bandit_with_known_answer():
    """Простейшая проверка, что обучение вообще работает.

    Однократный выбор из трёх действий, награды 0 / 1 / 0 — оптимум известен
    заранее. Если DQN не выучивает это, дело не в среде и не в гиперпараметрах.
    """
    algo = make_algo(obs_dim=1, num_actions=3, gamma=0.0, learning_rate=1e-2, batch_size=32)
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(1000, obs_dim=1, seed=0)

    rewards = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(200):
        actions = rng.integers(0, 3, size=8)
        buf.add_batch(
            obs=np.zeros((8, 1), dtype=np.float32),
            action=actions,
            reward=rewards[actions],
            next_obs=np.zeros((8, 1), dtype=np.float32),
            terminated=np.ones(8, dtype=bool),
            truncated=np.zeros(8, dtype=bool),
        )
        algo.update(buf.sample(32))

    assert int(algo.greedy_action(np.zeros((1, 1), dtype=np.float32))[0]) == 1


# --- сбор опыта ----------------------------------------------------------


def make_result(**kwargs) -> StepResult:
    n = len(kwargs["reward"])
    defaults = dict(
        obs=[np.zeros((n, 4), dtype=np.float32)],
        final_obs=[np.zeros((n, 4), dtype=np.float32)],
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        active=np.ones(n, dtype=bool),
        info={},
    )
    defaults.update(kwargs)
    return StepResult(**defaults)


def test_store_uses_final_obs_for_finished_episodes():
    buf = ReplayBuffer(10, obs_dim=4)
    result = make_result(
        reward=np.array([1.0], dtype=np.float32),
        obs=[np.full((1, 4), 9.0, dtype=np.float32)],       # уже новый эпизод
        final_obs=[np.full((1, 4), 7.0, dtype=np.float32)], # терминальное состояние
        terminated=np.array([True]),
    )
    _store_transitions(buf, np.zeros((1, 4), dtype=np.float32), np.array([2]), result)

    sample = buf.sample(1)
    assert np.allclose(sample.next_obs, 7.0)
    assert bool(sample.terminated[0])


def test_store_skips_slots_without_result():
    buf = ReplayBuffer(10, obs_dim=4)
    result = make_result(
        reward=np.array([0.0, 1.0], dtype=np.float32),
        active=np.array([False, True]),
    )
    added = _store_transitions(buf, np.zeros((2, 4), dtype=np.float32), np.array([0, 1]), result)

    assert added == 1
    assert int(buf.sample(1).action[0]) == 1


def test_entropy_matches_tabular_implementation():
    assert epsilon_greedy_entropy(1.0, 4) == pytest.approx(np.log(4))
    assert epsilon_greedy_entropy(0.0, 4) == pytest.approx(0.0)
