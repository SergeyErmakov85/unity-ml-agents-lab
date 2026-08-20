"""MA-POCA: внимание, контрфактический базлайн и групповой λ-возврат.

Свойства, которые здесь проверяются, — это ровно то, ради чего метод
устроен сложнее PPO. Каждое из них при поломке не даёт ни исключения,
ни расходящегося обучения: метод просто перестаёт быть многоагентным
и тихо вырождается в PPO с шумной целью.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from labrl.algos.mapoca import (
    MAPOCA,
    CentralizedCritic,
    CounterfactualBaseline,
    MAPOCAConfig,
)
from labrl.buffers.group_rollout import GroupRolloutBuffer
from labrl.nets.attention import ResidualSelfAttention, masked_mean
from labrl.nets.categorical_policy import MultiBranchCategoricalPolicy

OBS_DIM = 6
BRANCHES = (3, 3, 3)
TEAM = 3


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def make_algo(team: int = TEAM) -> MAPOCA:
    return MAPOCA(
        policy_net=MultiBranchCategoricalPolicy(OBS_DIM, BRANCHES, hidden_sizes=(16,)),
        value_net=CentralizedCritic(OBS_DIM, embed_dim=16, num_heads=2, hidden_dim=16),
        baseline_net=CounterfactualBaseline(
            OBS_DIM, BRANCHES, embed_dim=16, num_heads=2, hidden_dim=16
        ),
        cfg=MAPOCAConfig(minibatch_size=4, epochs=1),
        seed=0,
    )


# --- внимание -----------------------------------------------------------


def test_attention_is_permutation_equivariant():
    """Перестановка агентов переставляет выходы, но не меняет их значения.

    Без этого свойства «игрок 0 и игрок 1» и «игрок 1 и игрок 0» были бы
    для критика разными состояниями, и он учил бы каждую перестановку
    отдельно.
    """
    attention = ResidualSelfAttention(embed_dim=8, num_heads=2).eval()
    entities = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4)

    order = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        direct = attention(entities, mask)[0, order]
        permuted = attention(entities[:, order], mask)[0]

    assert torch.allclose(direct, permuted, atol=1e-5)


def test_attention_ignores_masked_entities():
    """Замаскированная сущность не влияет на представление существующих.

    Это условие «posthumous»: выбывший игрок исчезает из множества, а
    не превращается в агента, стоящего в нуле координат.
    """
    attention = ResidualSelfAttention(embed_dim=8, num_heads=2).eval()
    entities = torch.randn(1, 3, 8)
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    changed = entities.clone()
    changed[0, 2] = torch.randn(8) * 100.0

    with torch.no_grad():
        before = attention(entities, mask)[0, :2]
        after = attention(changed, mask)[0, :2]

    assert torch.allclose(before, after, atol=1e-5)


def test_masked_mean_uses_only_active():
    values = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert masked_mean(values, mask).item() == pytest.approx(2.0)


# --- контрфактический базлайн -------------------------------------------


def test_baseline_ignores_own_action_but_uses_others():
    """Главное свойство метода.

    ``Q_i`` обязан **не** зависеть от действия самого i-го игрока (иначе это
    не контрфактический базлайн, а обычный Q, и преимущество перестаёт
    измерять вклад) и **зависеть** от действий остальных (иначе базлайн
    выродился в V, и MA-POCA стал обычным PPO).
    """
    baseline = CounterfactualBaseline(OBS_DIM, BRANCHES, embed_dim=16, num_heads=2, hidden_dim=16).eval()
    obs = torch.randn(2, TEAM, OBS_DIM)
    action = torch.randint(0, 3, (2, TEAM, len(BRANCHES)))
    active = torch.ones(2, TEAM)

    with torch.no_grad():
        base = baseline(obs, action, active)

        own_changed = action.clone()
        own_changed[:, 0] = (own_changed[:, 0] + 1) % 3
        after_own = baseline(obs, own_changed, active)

        other_changed = action.clone()
        other_changed[:, 1] = (other_changed[:, 1] + 1) % 3
        after_other = baseline(obs, other_changed, active)

    # Q_0 не изменился от смены действия игрока 0…
    assert torch.allclose(base[:, 0], after_own[:, 0], atol=1e-6)
    # …но изменился от смены действия игрока 1.
    assert not torch.allclose(base[:, 0], after_other[:, 0], atol=1e-4)
    # А Q_1, наоборот, к собственному действию игрока 1 безразличен.
    assert torch.allclose(base[:, 1], after_other[:, 1], atol=1e-6)


def test_baseline_output_shape():
    baseline = CounterfactualBaseline(OBS_DIM, BRANCHES, embed_dim=16, num_heads=2, hidden_dim=16)
    out = baseline(
        torch.randn(5, TEAM, OBS_DIM),
        torch.randint(0, 3, (5, TEAM, len(BRANCHES))),
        torch.ones(5, TEAM),
    )
    assert out.shape == (5, TEAM)


def test_critic_is_permutation_invariant():
    """V команды не зависит от порядка игроков."""
    critic = CentralizedCritic(OBS_DIM, embed_dim=16, num_heads=2, hidden_dim=16).eval()
    obs = torch.randn(1, TEAM, OBS_DIM)
    active = torch.ones(1, TEAM)
    order = torch.tensor([2, 0, 1])

    with torch.no_grad():
        assert torch.allclose(critic(obs, active), critic(obs[:, order], active), atol=1e-5)


# --- групповой буфер ----------------------------------------------------


def add_step(buffer: GroupRolloutBuffer, reward: float, value: float,
             terminated: bool = False, truncated: bool = False, bootstrap: float = 0.0) -> None:
    buffer.add(
        group=0,
        obs=np.zeros((TEAM, OBS_DIM), dtype=np.float32),
        action=np.zeros((TEAM, len(BRANCHES)), dtype=np.int64),
        active=np.ones(TEAM, dtype=np.float32),
        log_prob=np.zeros(TEAM, dtype=np.float32),
        value=value,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        bootstrap_value=bootstrap,
    )


def test_group_buffer_terminated_has_no_future():
    """Истинное завершение: λ-возврат равен самой награде."""
    buffer = GroupRolloutBuffer(num_groups=1, gamma=0.9, gae_lambda=0.8)
    add_step(buffer, reward=2.0, value=0.5, terminated=True)
    batch = buffer.compute(np.zeros(1))
    assert batch.returns[0] == pytest.approx(2.0)


def test_group_buffer_truncated_bootstraps():
    """Обрыв по времени: будущее подставляется из критика (8.3).

    Эта строка отличает «время вышло» от «всё потеряно». Без неё команда
    выучила бы, что тянуть время — катастрофа, и начала бы рисковать
    там, где риск не нужен.
    """
    buffer = GroupRolloutBuffer(num_groups=1, gamma=0.9, gae_lambda=0.8)
    add_step(buffer, reward=1.0, value=0.5, truncated=True, bootstrap=4.0)
    batch = buffer.compute(np.zeros(1))
    # G = r + γ·V(s_последнее) = 1 + 0.9·4 = 4.6
    assert batch.returns[0] == pytest.approx(4.6)


def test_group_buffer_lambda_return_matches_manual_sum():
    """Свёртка TD(λ) на двух шагах, посчитанная руками."""
    gamma, lam = 0.9, 0.8
    buffer = GroupRolloutBuffer(num_groups=1, gamma=gamma, gae_lambda=lam)
    add_step(buffer, reward=1.0, value=0.5)
    add_step(buffer, reward=2.0, value=1.0, terminated=True)
    batch = buffer.compute(np.zeros(1))

    delta1 = 2.0 - 1.0
    delta0 = 1.0 + gamma * 1.0 - 0.5
    advantage0 = delta0 + gamma * lam * delta1
    assert batch.returns[1] == pytest.approx(1.0 + delta1)
    assert batch.returns[0] == pytest.approx(advantage0 + 0.5)


# --- алгоритм в сборе ---------------------------------------------------


def test_update_changes_parameters_and_reports_stats():
    algo = make_algo()
    buffer = GroupRolloutBuffer(num_groups=2, gamma=0.99, gae_lambda=0.95)
    rng = np.random.default_rng(0)

    for group in range(2):
        for t in range(8):
            buffer.add(
                group=group,
                obs=rng.normal(size=(TEAM, OBS_DIM)).astype(np.float32),
                action=rng.integers(0, 3, size=(TEAM, len(BRANCHES))).astype(np.int64),
                active=np.ones(TEAM, dtype=np.float32),
                log_prob=rng.normal(size=TEAM).astype(np.float32),
                value=float(rng.normal()),
                reward=float(rng.normal()),
                terminated=t == 7,
                truncated=False,
            )

    before = [p.detach().clone() for p in algo.policy_net.parameters()]
    stats = algo.update(buffer.compute(np.zeros(2)))
    after = list(algo.policy_net.parameters())

    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    for key in ("policy_loss", "value_loss", "baseline_loss", "entropy", "baseline_spread"):
        assert key in stats and np.isfinite(stats[key])


def test_inactive_players_do_not_affect_the_loss():
    """Отсутствующий игрок не должен вносить вклад в функцию потерь.

    Если маска не применена, «пустой» игрок с нулевым наблюдением и нулевым
    действием попадает в среднее и разбавляет градиент.
    """
    def batch_with(second_player_obs: np.ndarray):
        # Свой генератор на каждый вызов: иначе наблюдения ЖИВОГО игрока
        # различались бы между двумя батчами, и тест ловил бы это различие,
        # а не влияние отсутствующего игрока.
        rng = np.random.default_rng(1)
        buffer = GroupRolloutBuffer(num_groups=1, gamma=0.99, gae_lambda=0.95)
        for t in range(4):
            obs = np.zeros((TEAM, OBS_DIM), dtype=np.float32)
            obs[0] = rng.normal(size=OBS_DIM)
            obs[1] = second_player_obs
            active = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            buffer.add(
                group=0,
                obs=obs,
                action=np.zeros((TEAM, len(BRANCHES)), dtype=np.int64),
                active=active,
                log_prob=np.zeros(TEAM, dtype=np.float32),
                value=0.0,
                reward=1.0,
                terminated=t == 3,
                truncated=False,
            )
        return buffer.compute(np.zeros(1))

    torch.manual_seed(7)
    algo_a = make_algo()
    stats_a = algo_a.update(batch_with(np.zeros(OBS_DIM, dtype=np.float32)))

    torch.manual_seed(7)
    algo_b = make_algo()
    stats_b = algo_b.update(batch_with(np.full(OBS_DIM, 50.0, dtype=np.float32)))

    assert stats_a["policy_loss"] == pytest.approx(stats_b["policy_loss"], abs=1e-6)


def test_policy_module_is_the_actor_only():
    """В ONNX уходит только актор: у игрока в Unity нет наблюдений партнёра."""
    algo = make_algo()
    assert algo.policy_module() is algo.policy_net


def test_act_returns_indices_within_branches():
    algo = make_algo()
    action, log_prob = algo.act(np.zeros((5, OBS_DIM), dtype=np.float32))
    assert action.shape == (5, len(BRANCHES))
    assert log_prob.shape == (5,)
    for branch, size in enumerate(BRANCHES):
        assert action[:, branch].min() >= 0 and action[:, branch].max() < size


def test_state_dict_round_trip():
    algo = make_algo()
    other = make_algo()
    other.load_state_dict(algo.state_dict())
    obs = np.zeros((3, OBS_DIM), dtype=np.float32)
    assert np.array_equal(algo.deterministic_action(obs), other.deterministic_action(obs))
