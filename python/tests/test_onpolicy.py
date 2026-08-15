"""Непрерывное управление: гауссова политика, A2C и PPO (`E04_BallBalance`).

Главный тест файла — `test_deterministic_action_matches_onnx_graph`: он
проверяет то самое место, где Python и Unity расходятся чаще всего. Оценка
в Python и инференс в Unity обязаны вычислять **одно и то же** действие,
включая приведение к диапазону `clamp(x, −3, 3) / 3` из контракта ML-Agents.
"""

from __future__ import annotations

import math

import numpy as np
import onnxruntime as ort
import pytest
import torch

from labrl.algos.a2c import A2C, A2CConfig, explained_variance
from labrl.algos.ppo import PPO, PPOConfig
from labrl.buffers.rollout import RolloutBuffer
from labrl.export.onnx_export import ActionSpecLite, export_policy_to_onnx
from labrl.export.onnx_verify import verify_onnx_model
from labrl.nets.gaussian_policy import GaussianPolicyNetwork, to_env_action
from labrl.nets.mlp import MLPValueNetwork

OBS_DIM = 4
ACTION_DIM = 2


def make_algo(kind: str = "a2c", **kwargs):
    torch.manual_seed(0)
    policy = GaussianPolicyNetwork(OBS_DIM, ACTION_DIM, (16, 16), "tanh", log_std_init=-0.5)
    value = MLPValueNetwork(OBS_DIM, (16, 16), "tanh")
    if kind == "a2c":
        return A2C(policy, value, A2CConfig(**kwargs))
    return PPO(policy, value, PPOConfig(**kwargs), seed=0)


def make_batch(algo, size: int = 64, seed: int = 0):
    """Собирает роллаут случайными наблюдениями через настоящий буфер."""
    rng = np.random.default_rng(seed)
    buffer = RolloutBuffer(num_envs=1, gamma=algo.cfg.gamma, gae_lambda=algo.cfg.gae_lambda)
    obs = rng.normal(size=(1, OBS_DIM)).astype(np.float32)
    for step in range(size):
        out = algo.act(obs, rng)
        terminated = step % 17 == 16
        buffer.add(
            slot=0,
            obs=obs[0],
            action=out.raw_action[0],
            log_prob=float(out.log_prob[0]),
            value=float(out.value[0]),
            reward=float(rng.normal()),
            terminated=terminated,
            truncated=False,
        )
        obs = rng.normal(size=(1, OBS_DIM)).astype(np.float32)
    return buffer.compute(last_values=algo.value(obs))


# --- приведение действия к диапазону Unity ------------------------------


def test_to_env_action_scales_and_clips():
    raw = np.array([[-9.0, -3.0, 0.0, 1.5, 9.0]])
    np.testing.assert_allclose(to_env_action(raw), np.array([[-1.0, -1.0, 0.0, 0.5, 1.0]]))


def test_to_env_action_works_for_torch_and_numpy_identically():
    raw = np.array([[-4.0, 0.3]], dtype=np.float32)
    np.testing.assert_allclose(
        to_env_action(torch.from_numpy(raw)).numpy(), to_env_action(raw), rtol=1e-6
    )


# --- гауссова политика --------------------------------------------------


def test_log_prob_matches_normal_density():
    """log π(a|s) — сумма по компонентам, а не среднее: компоненты независимы."""
    torch.manual_seed(0)
    net = GaussianPolicyNetwork(OBS_DIM, ACTION_DIM, (8,), "tanh", log_std_init=-0.5)
    obs = torch.randn(5, OBS_DIM)
    action = torch.randn(5, ACTION_DIM)

    mean = net(obs)
    std = net.clamped_log_std().exp()
    manual = torch.distributions.Normal(mean, std).log_prob(action).sum(dim=-1)

    torch.testing.assert_close(net.log_prob(obs, action), manual)


def test_entropy_matches_closed_form():
    net = GaussianPolicyNetwork(OBS_DIM, ACTION_DIM, (8,), "tanh", log_std_init=-0.5)
    expected = ACTION_DIM * (-0.5 + 0.5 * math.log(2.0 * math.pi * math.e))
    assert net.entropy().item() == pytest.approx(expected, rel=1e-6)


def test_log_std_is_clamped():
    """σ не должна вырождаться в точку: градиент log π при σ → 0 расходится."""
    net = GaussianPolicyNetwork(OBS_DIM, ACTION_DIM, (8,), "tanh", log_std_init=-50.0)
    assert net.clamped_log_std().min().item() >= -4.0


# --- совпадение Python и графа ONNX -------------------------------------


def test_deterministic_action_matches_onnx_graph(tmp_path):
    """Оценка в Python и инференс в Unity обязаны давать одно и то же действие.

    Проверяется сквозь настоящий ONNX-граф: расхождение здесь — типовая
    причина провала требования 10.6 «награда в Unity ≥ 0.8 от Python».
    """
    algo = make_algo("a2c")
    spec = ActionSpecLite(continuous_size=ACTION_DIM)
    path = export_policy_to_onnx(algo.policy_module(), spec, [(OBS_DIM,)], tmp_path / "policy.onnx")

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(16, OBS_DIM)).astype(np.float32)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    graph_action = session.run(["deterministic_continuous_actions"], {"obs_0": obs})[0]

    np.testing.assert_allclose(algo.deterministic_action(obs), graph_action, rtol=0, atol=1e-6)


def test_continuous_policy_passes_onnx_contract(tmp_path):
    algo = make_algo("ppo", clip_range=0.2, epochs=1, minibatch_size=32)
    spec = ActionSpecLite(continuous_size=ACTION_DIM)
    path = export_policy_to_onnx(algo.policy_module(), spec, [(OBS_DIM,)], tmp_path / "policy.onnx")

    rng = np.random.default_rng(1)
    sample = [rng.normal(size=(64, OBS_DIM)).astype(np.float32)]
    result = verify_onnx_model(path, algo.policy_module(), spec, [(OBS_DIM,)], sample_obs=sample)

    assert result.passed, result.report()


def test_sampled_action_stays_in_unity_range():
    algo = make_algo("a2c")
    rng = np.random.default_rng(0)
    out = algo.act(rng.normal(size=(32, OBS_DIM)).astype(np.float32), rng)
    assert np.all(np.abs(out.env_action) <= 1.0 + 1e-6)


def test_act_is_reproducible_for_the_same_generator():
    """Разведочный шум берётся из переданного генератора, а не из глобального."""
    algo = make_algo("a2c")
    obs = np.zeros((4, OBS_DIM), dtype=np.float32)
    first = algo.act(obs, np.random.default_rng(7)).raw_action
    second = algo.act(obs, np.random.default_rng(7)).raw_action
    np.testing.assert_allclose(first, second)


# --- обновление ---------------------------------------------------------


def test_a2c_update_changes_parameters_and_reports_metrics():
    algo = make_algo("a2c")
    before = [p.detach().clone() for p in algo.policy_net.parameters()]

    metrics = algo.update(make_batch(algo))

    assert set(metrics) >= {"loss", "policy_loss", "value_loss", "entropy", "explained_variance"}
    assert any(not torch.equal(b, p) for b, p in zip(before, algo.policy_net.parameters()))


def test_a2c_increases_probability_of_advantageous_action():
    """Проверка знака градиента: действие с положительным преимуществом
    после шага обязано стать вероятнее.
    """
    algo = make_algo("a2c", entropy_coef=0.0, normalize_advantage=False, learning_rate=0.05)
    obs = np.zeros((1, OBS_DIM), dtype=np.float32)
    action = np.full((1, ACTION_DIM), 0.5, dtype=np.float32)

    obs_t = torch.from_numpy(obs)
    action_t = torch.from_numpy(action)
    before = algo.policy_net.log_prob(obs_t, action_t).item()

    from labrl.buffers.rollout import RolloutBatch

    algo.update(
        RolloutBatch(
            obs=obs, action=action,
            log_prob=np.array([before], dtype=np.float32),
            advantage=np.array([1.0], dtype=np.float32),
            returns=np.array([0.0], dtype=np.float32),
            value=np.array([0.0], dtype=np.float32),
        )
    )

    after = algo.policy_net.log_prob(obs_t, action_t).item()
    assert after > before


def test_ppo_runs_several_epochs_and_reports_clip_fraction():
    algo = make_algo("ppo", clip_range=0.2, epochs=3, minibatch_size=16)
    metrics = algo.update(make_batch(algo))

    assert metrics["epochs_done"] == 3
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["approx_kl"] >= 0.0


def test_ppo_target_kl_stops_epochs_early():
    """Страховка от слишком большого шага: эпохи прекращаются досрочно."""
    algo = make_algo("ppo", clip_range=0.2, epochs=8, minibatch_size=16,
                     learning_rate=0.5, target_kl=1e-6)
    metrics = algo.update(make_batch(algo))
    assert metrics["epochs_done"] < 8


def test_ppo_clipping_limits_the_step():
    """При одинаковых данных PPO с обрезкой уходит от старой политики
    не дальше, чем без неё.
    """
    batch_source = make_algo("a2c")
    batch = make_batch(batch_source)

    tight = make_algo("ppo", clip_range=0.01, epochs=4, minibatch_size=32, learning_rate=0.01)
    loose = make_algo("ppo", clip_range=10.0, epochs=4, minibatch_size=32, learning_rate=0.01)

    tight_kl = tight.update(batch)["approx_kl"]
    loose_kl = loose.update(batch)["approx_kl"]
    assert tight_kl <= loose_kl + 1e-9


# --- диагностика критика ------------------------------------------------


def test_explained_variance_is_one_for_perfect_prediction():
    target = np.array([1.0, 2.0, 3.0])
    assert explained_variance(target, target) == pytest.approx(1.0)


def test_explained_variance_is_zero_for_constant_prediction():
    target = np.array([1.0, 2.0, 3.0])
    assert explained_variance(np.full(3, target.mean()), target) == pytest.approx(0.0)


def test_explained_variance_of_constant_target_is_zero():
    """Вырожденный случай не должен делить на ноль."""
    assert explained_variance(np.zeros(3), np.ones(3)) == 0.0


# --- сохранение ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["a2c", "ppo"])
def test_state_dict_round_trip(kind):
    algo = make_algo(kind) if kind == "a2c" else make_algo(kind, clip_range=0.2, epochs=1, minibatch_size=32)
    algo.update(make_batch(algo))

    restored = make_algo(kind) if kind == "a2c" else make_algo(kind, clip_range=0.2, epochs=1, minibatch_size=32)
    restored.load_state_dict(algo.state_dict())

    obs = np.random.default_rng(0).normal(size=(8, OBS_DIM)).astype(np.float32)
    np.testing.assert_allclose(restored.deterministic_action(obs), algo.deterministic_action(obs))
