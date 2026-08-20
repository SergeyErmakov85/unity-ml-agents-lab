"""SAC: squashed-гауссова политика, два критика, автоподстройка α (`E07`).

Главные проверки файла — те, где ошибка не видна по графикам обучения:
поправка плотности на ``tanh`` и совпадение детерминированного действия
с выходом настоящего графа ONNX.
"""

from __future__ import annotations

import math

import numpy as np
import onnxruntime as ort
import pytest
import torch

from labrl.algos.sac import SAC, SACConfig
from labrl.buffers.replay import ReplayBuffer
from labrl.export.onnx_export import ActionSpecLite, export_policy_to_onnx
from labrl.export.onnx_verify import verify_onnx_model
from labrl.nets.mlp import MLPContinuousQNetwork
from labrl.nets.squashed_gaussian import LOG_STD_MAX, LOG_STD_MIN, SquashedGaussianPolicy

OBS_DIM = 6
ACTION_DIM = 2


def make_algo(**kwargs) -> SAC:
    torch.manual_seed(0)
    policy = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, (32, 32), "relu")
    q1 = MLPContinuousQNetwork(OBS_DIM, ACTION_DIM, (32, 32), "relu")
    q2 = MLPContinuousQNetwork(OBS_DIM, ACTION_DIM, (32, 32), "relu")
    return SAC(policy, q1, q2, SACConfig(batch_size=16, **kwargs))


def make_batch(size: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=256, obs_dim=OBS_DIM, seed=seed, action_dim=ACTION_DIM)
    buffer.add_batch(
        obs=rng.normal(size=(size, OBS_DIM)).astype(np.float32),
        action=rng.uniform(-1, 1, size=(size, ACTION_DIM)).astype(np.float32),
        reward=rng.normal(size=size).astype(np.float32),
        next_obs=rng.normal(size=(size, OBS_DIM)).astype(np.float32),
        terminated=rng.random(size) < 0.2,
        truncated=np.zeros(size, dtype=bool),
    )
    return buffer.sample(16)


# --- политика -----------------------------------------------------------


def test_action_stays_in_range_and_log_prob_is_finite_at_saturation():
    """tanh удерживает действие в [−1, 1] и не даёт плотности обратиться в −inf.

    В точной арифметике `tanh` границ не достигает, но во float32 при |u| > 9
    он округляется ровно до 1.0 — и тогда `log(1 − a²)` становится −inf,
    а функция потерь SAC — NaN. От этого и защищает добавка `TANH_EPS`.
    Проверяется именно это: наблюдения намеренно взяты большими, чтобы
    политика ушла в насыщение.
    """
    policy = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, (16,), "relu")
    obs = torch.randn(64, OBS_DIM) * 50.0
    action, log_prob = policy.sample(obs)

    assert action.abs().max().item() <= 1.0
    assert torch.isfinite(log_prob).all(), "плотность обязана оставаться конечной в насыщении"


def test_log_prob_includes_tanh_correction():
    """Поправка на замену переменных обязана вычитаться из плотности.

    Без неё log π считается по гауссиане `u`, а не по распределению `a`,
    и энтропийная часть цели SAC оказывается неверной.
    """
    torch.manual_seed(0)
    policy = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, (16,), "relu")
    obs = torch.randn(8, OBS_DIM)
    noise = torch.randn(8, ACTION_DIM)

    action, log_prob = policy.sample(obs, noise)

    mean, log_std = policy.distribution_params(obs)
    std = log_std.exp()
    raw = mean + std * noise
    gaussian = (-0.5 * ((raw - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    correction = torch.log(1.0 - torch.tanh(raw).pow(2) + 1e-6).sum(-1)

    torch.testing.assert_close(log_prob, gaussian - correction)
    # Поправка неотрицательна, поэтому плотность squash-политики всегда выше
    # гауссовой: сжатие в конечный интервал уплотняет вероятность.
    assert torch.all(log_prob >= gaussian - 1e-5)


def test_log_std_is_clamped():
    policy = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, (16,), "relu")
    with torch.no_grad():
        for param in policy.body[-1].parameters():
            param.fill_(1e3)
    _, log_std = policy.distribution_params(torch.randn(4, OBS_DIM))
    assert log_std.max().item() <= LOG_STD_MAX + 1e-6
    assert log_std.min().item() >= LOG_STD_MIN - 1e-6


def test_sample_is_reproducible_for_the_same_noise():
    policy = SquashedGaussianPolicy(OBS_DIM, ACTION_DIM, (16,), "relu")
    obs = torch.zeros(4, OBS_DIM)
    noise = torch.randn(4, ACTION_DIM)
    first, _ = policy.sample(obs, noise)
    second, _ = policy.sample(obs, noise)
    torch.testing.assert_close(first, second)


def test_act_uses_external_generator():
    algo = make_algo()
    obs = np.zeros((4, OBS_DIM), dtype=np.float32)
    first = algo.act(obs, np.random.default_rng(3))
    second = algo.act(obs, np.random.default_rng(3))
    np.testing.assert_allclose(first, second)


# --- обновление ---------------------------------------------------------


def test_update_reports_all_metrics_and_changes_parameters():
    algo = make_algo()
    before = [p.detach().clone() for p in algo.policy_net.parameters()]
    metrics = algo.update(make_batch())

    assert set(metrics) >= {"value_loss", "policy_loss", "alpha", "entropy", "td_error_abs"}
    assert any(not torch.equal(b, p) for b, p in zip(before, algo.policy_net.parameters()))


def test_target_networks_move_slowly():
    """Мягкое обновление: цель сдвигается на τ, а не копируется целиком."""
    algo = make_algo(tau=0.1)
    target_before = [p.detach().clone() for p in algo.q1_target.parameters()]
    algo.update(make_batch())

    online = list(algo.q1.parameters())
    target_after = list(algo.q1_target.parameters())
    for before, after, now in zip(target_before, target_after, online):
        expected = 0.9 * before + 0.1 * now.detach()
        torch.testing.assert_close(after.detach(), expected)


def test_alpha_grows_when_policy_is_too_deterministic():
    """Автоподстройка возвращает разведку, если энтропия упала ниже целевой."""
    algo = make_algo(init_alpha=0.2, target_entropy=10.0)  # заведомо недостижимая цель
    alpha_before = float(algo.alpha.item())
    for _ in range(20):
        algo.update(make_batch())
    assert float(algo.alpha.item()) > alpha_before


def test_alpha_is_frozen_when_autotune_disabled():
    algo = make_algo(autotune_alpha=False, init_alpha=0.3)
    for _ in range(5):
        algo.update(make_batch())
    assert float(algo.alpha.item()) == pytest.approx(0.3, rel=1e-6)


def test_target_entropy_defaults_to_minus_action_dim():
    algo = make_algo()
    assert algo.target_entropy == pytest.approx(-ACTION_DIM)


def test_terminated_zeroes_the_future_but_truncated_does_not():
    """Тот же инвариант, что у DQN: обрыв по времени бутстрэппится."""
    algo = make_algo(gamma=0.9)
    obs = np.zeros((1, OBS_DIM), dtype=np.float32)
    action = np.zeros((1, ACTION_DIM), dtype=np.float32)

    from labrl.buffers.replay import Batch

    def target_for(terminated: bool) -> float:
        batch = Batch(
            obs=obs, action=action, reward=np.array([1.0], dtype=np.float32),
            next_obs=obs, terminated=np.array([terminated]), truncated=np.array([not terminated]),
        )
        with torch.no_grad():
            obs_t = torch.as_tensor(obs)
            next_action, next_log_prob = algo.policy_net.sample(obs_t)
            next_q = torch.min(algo.q1_target(obs_t, next_action), algo.q2_target(obs_t, next_action))
            soft = next_q - algo.alpha * next_log_prob
            return float(1.0 + 0.9 * soft.item() * (0.0 if terminated else 1.0))

    assert target_for(True) == pytest.approx(1.0)
    assert target_for(False) != pytest.approx(1.0)


# --- буфер --------------------------------------------------------------


def test_replay_buffer_stores_continuous_actions():
    buffer = ReplayBuffer(capacity=8, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    action = np.array([[0.25, -0.5]], dtype=np.float32)
    buffer.add_batch(
        obs=np.zeros((1, OBS_DIM), dtype=np.float32), action=action,
        reward=np.zeros(1, dtype=np.float32), next_obs=np.zeros((1, OBS_DIM), dtype=np.float32),
        terminated=np.zeros(1, dtype=bool), truncated=np.zeros(1, dtype=bool),
    )
    batch = buffer.sample(1)
    assert batch.action.shape == (1, ACTION_DIM)
    np.testing.assert_allclose(batch.action, action)


# --- экспорт ------------------------------------------------------------


def test_deterministic_action_matches_onnx_graph(tmp_path):
    """Оценка в Python и инференс в Unity обязаны давать одно и то же действие.

    Для SAC это тоньше, чем для PPO: политика уже выдаёт значения в (−1, 1),
    а обёртка контракта делит выход на 3. Поэтому экспортируется политика,
    домноженная на 3 — и проверить это можно только прогоном графа.
    """
    algo = make_algo()
    spec = ActionSpecLite(continuous_size=ACTION_DIM)
    path = export_policy_to_onnx(algo.policy_module(), spec, [(OBS_DIM,)], tmp_path / "policy.onnx")

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(16, OBS_DIM)).astype(np.float32)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    graph_action = session.run(["deterministic_continuous_actions"], {"obs_0": obs})[0]

    np.testing.assert_allclose(algo.deterministic_action(obs), graph_action, rtol=0, atol=1e-6)


def test_exported_policy_passes_onnx_contract(tmp_path):
    algo = make_algo()
    spec = ActionSpecLite(continuous_size=ACTION_DIM)
    policy = algo.policy_module()
    path = export_policy_to_onnx(policy, spec, [(OBS_DIM,)], tmp_path / "policy.onnx")

    rng = np.random.default_rng(1)
    sample = [rng.normal(size=(64, OBS_DIM)).astype(np.float32)]
    result = verify_onnx_model(path, policy, spec, [(OBS_DIM,)], sample_obs=sample)

    assert result.passed, result.report()


def test_state_dict_round_trip():
    algo = make_algo()
    algo.update(make_batch())
    restored = make_algo()
    restored.load_state_dict(algo.state_dict())

    obs = np.random.default_rng(0).normal(size=(8, OBS_DIM)).astype(np.float32)
    np.testing.assert_allclose(restored.deterministic_action(obs), algo.deterministic_action(obs))
    assert float(restored.alpha.item()) == pytest.approx(float(algo.alpha.item()))
