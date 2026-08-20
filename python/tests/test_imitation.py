"""Имитационное обучение: эксперт, демонстрации, BC и дискриминатор GAIL.

Главная проверяемая вещь — **эксперт действует только по наблюдению агента**.
Если он пользуется знанием, недоступному агенту, имитация учится
невозможному: политика идеально повторяет эксперта на записях и разваливается
в среде, потому что нужного признака во входе просто нет. Такую ошибку
не видно ни в одной метрике обучения.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from labrl.algos.bc import BC, BCConfig
from labrl.algos.gail import GAIL, Discriminator, GAILConfig
from labrl.envs.demos import Demonstrations, corridor_expert
from labrl.nets.categorical_policy import MultiBranchCategoricalPolicy

OBS_DIM = 13
BRANCHES = (4,)
GRID = 9

#: Индексы «щупалец» в наблюдении `E10_Imitation`: север, юг, восток, запад.
WALLS = slice(7, 11)
#: Индекс направления предыдущего хода / 3.
PREVIOUS = 12


# --- модель среды для теста ---------------------------------------------


def is_wall(x: int, y: int) -> bool:
    """Та же раскладка, что в `CorridorArea.IsWallCell` (TS-010, §3)."""
    if x < 0 or y < 0 or x >= GRID or y >= GRID:
        return True
    if y % 2 == 0:
        return False
    gap = GRID - 1 if (y // 2) % 2 == 0 else 0
    return x != gap


def observation(x: int, y: int, previous: int) -> np.ndarray:
    """Наблюдение агента в клетке (x, y) после хода `previous` (−1 в начале)."""
    scale = GRID - 1
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0], obs[1] = x / scale, y / scale
    obs[2], obs[3] = 1.0, 1.0
    obs[4], obs[5] = (scale - x) / scale, (scale - y) / scale
    obs[6] = (abs(scale - x) + abs(scale - y)) / (2 * scale)
    obs[7] = 1.0 if is_wall(x, y + 1) else 0.0
    obs[8] = 1.0 if is_wall(x, y - 1) else 0.0
    obs[9] = 1.0 if is_wall(x + 1, y) else 0.0
    obs[10] = 1.0 if is_wall(x - 1, y) else 0.0
    obs[11] = 0.0
    obs[PREVIOUS] = previous / 3.0
    return obs


MOVES = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}


def run_expert(max_steps: int = 200) -> tuple[int, tuple[int, int]]:
    """Прогоняет эксперта по коридору. Возвращает число ходов и конечную клетку."""
    x, y, previous = 0, 0, -1
    for step in range(max_steps):
        if (x, y) == (GRID - 1, GRID - 1):
            return step, (x, y)
        action = int(corridor_expert(observation(x, y, previous)[None, :])[0, 0])
        dx, dy = MOVES[action]
        if not is_wall(x + dx, y + dy):
            x, y = x + dx, y + dy
        previous = action
    return max_steps, (x, y)


# --- эксперт ------------------------------------------------------------


def test_expert_reaches_goal_in_shortest_path():
    """Эксперт проходит коридор ровно за 48 ходов — длину кратчайшего пути.

    Больше — значит правило где-то петляет; меньше невозможно.
    """
    steps, cell = run_expert()
    assert cell == (GRID - 1, GRID - 1), f"эксперт застрял в {cell}"
    assert steps == 48, f"эксперт прошёл за {steps} ходов вместо 48"


def test_expert_uses_only_the_agent_observation():
    """Эксперт обязан читать только «щупальца» и предыдущий ход.

    Проверяется тем, что зашумление всех ОСТАЛЬНЫХ признаков не меняет
    его решения. Если меняет — эксперт опирается на что-то ещё, и часть
    его знания в наблюдении агента отсутствует либо избыточна.
    """
    rng = np.random.default_rng(0)
    base = observation(3, 0, 2)[None, :]
    action = corridor_expert(base)

    for _ in range(20):
        noisy = base.copy()
        for index in range(OBS_DIM):
            if index in range(WALLS.start, WALLS.stop) or index == PREVIOUS:
                continue
            noisy[0, index] = rng.normal()
        assert np.array_equal(corridor_expert(noisy), action)


def test_expert_does_not_step_into_walls():
    """Ни одно решение эксперта не должно упираться в стену."""
    x, y, previous = 0, 0, -1
    for _ in range(48):
        action = int(corridor_expert(observation(x, y, previous)[None, :])[0, 0])
        dx, dy = MOVES[action]
        assert not is_wall(x + dx, y + dy), f"эксперт пошёл в стену из ({x}, {y})"
        x, y, previous = x + dx, y + dy, action


def test_expert_is_deterministic():
    """Два вызова на одном наблюдении обязаны давать одно действие:
    иначе два запуска дают разные демонстрации."""
    obs = observation(4, 2, 3)[None, :]
    assert np.array_equal(corridor_expert(obs), corridor_expert(obs))


def test_expert_rejects_short_observation():
    with pytest.raises(ValueError, match=">=13"):
        corridor_expert(np.zeros((2, 5), dtype=np.float32))


# --- демонстрации -------------------------------------------------------


def make_demos(n: int = 200) -> Demonstrations:
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, OBS_DIM)).astype(np.float32)
    action = rng.integers(0, 4, size=(n, 1)).astype(np.int64)
    return Demonstrations(obs, action, episodes=10, success_rate=1.0)


def test_split_is_disjoint_and_complete():
    demos = make_demos(100)
    train, holdout = demos.split(0.2, np.random.default_rng(0))
    assert len(train) + len(holdout) == len(demos)
    assert len(holdout) == 20


def test_split_rejects_degenerate_holdout():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        make_demos().split(0.0, np.random.default_rng(0))


def test_save_load_round_trip(tmp_path):
    demos = make_demos(50)
    path = demos.save(tmp_path / "demos.npz")
    restored = Demonstrations.load(path)
    assert np.array_equal(demos.obs, restored.obs)
    assert np.array_equal(demos.action, restored.action)
    assert restored.episodes == demos.episodes


# --- BC -----------------------------------------------------------------


def test_bc_learns_a_deterministic_mapping():
    """BC обязан выучить простое правило до высокой точности.

    Правило: действие = индекс минимального из первых четырёх признаков.
    Если BC не справляется с этим, дело не в среде.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(2000, OBS_DIM)).astype(np.float32)
    action = np.argmin(obs[:, :4], axis=1)[:, None].astype(np.int64)
    demos = Demonstrations(obs, action)

    algo = BC(
        MultiBranchCategoricalPolicy(OBS_DIM, BRANCHES, hidden_sizes=(64, 64)),
        BCConfig(learning_rate=3e-3, batch_size=256),
        seed=0,
    )
    algo.fit(demos, steps=600)
    assert algo.accuracy(demos) > 0.9


def test_bc_update_reports_accuracy_and_changes_parameters():
    torch.manual_seed(0)
    algo = BC(MultiBranchCategoricalPolicy(OBS_DIM, BRANCHES, hidden_sizes=(16,)), BCConfig())
    before = [p.detach().clone() for p in algo.policy_net.parameters()]
    stats = algo.update(make_demos().sample(64, np.random.default_rng(0)))
    after = list(algo.policy_net.parameters())

    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert 0.0 <= stats["accuracy"] <= 1.0
    assert np.isfinite(stats["policy_loss"])


def test_bc_policy_module_is_the_network():
    algo = BC(MultiBranchCategoricalPolicy(OBS_DIM, BRANCHES, hidden_sizes=(16,)))
    assert algo.policy_module() is algo.policy_net


# --- GAIL ---------------------------------------------------------------


def make_gail(cfg: GAILConfig | None = None) -> GAIL:
    torch.manual_seed(0)
    return GAIL(
        Discriminator(OBS_DIM, BRANCHES, hidden_sizes=(32, 32)),
        make_demos(),
        cfg or GAILConfig(batch_size=32, updates_per_batch=1, gradient_penalty=0.0),
        seed=0,
    )


def test_gail_reward_is_positive_and_monotone_in_the_logit():
    """r = −log(1 − D) = softplus(logit): положительна и растёт с логитом.

    Положительность — не косметика: именно из-за неё GAIL склонен тянуть
    эпизод, и именно поэтому существует env_reward_weight.
    """
    gail = make_gail()
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
    action = rng.integers(0, 4, size=(64, 1)).astype(np.int64)

    reward = gail.reward(obs, action)
    assert reward.shape == (64,)
    assert (reward > 0).all()

    with torch.no_grad():
        logits = gail.discriminator(
            torch.as_tensor(obs), torch.as_tensor(action)
        ).numpy()
    # Порядок наград обязан совпадать с порядком логитов.
    assert np.array_equal(np.argsort(reward), np.argsort(logits))


def test_gail_reward_matches_the_closed_form():
    """softplus(x) обязан совпадать с −log(1 − σ(x)) там, где прямая формула
    ещё считается устойчиво."""
    gail = make_gail()
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(16, OBS_DIM)).astype(np.float32)
    action = rng.integers(0, 4, size=(16, 1)).astype(np.int64)

    with torch.no_grad():
        logits = gail.discriminator(torch.as_tensor(obs), torch.as_tensor(action))
        direct = -torch.log1p(-torch.sigmoid(logits)).numpy()
    assert np.allclose(gail.reward(obs, action), direct, atol=1e-5)


def test_mixed_reward_endpoints():
    """Вес 0 — чистый GAIL, вес 1 — чистый RL."""
    gail_reward = np.array([1.0, 2.0])
    env_reward = np.array([10.0, 20.0])

    pure = make_gail(GAILConfig(env_reward_weight=0.0, gradient_penalty=0.0))
    assert np.allclose(pure.mixed_reward(gail_reward, env_reward), gail_reward)

    only_env = make_gail(GAILConfig(env_reward_weight=1.0, gradient_penalty=0.0))
    assert np.allclose(only_env.mixed_reward(gail_reward, env_reward), env_reward)


def test_discriminator_separates_expert_from_noise():
    """После обучения дискриминатор обязан отличать эксперта от чужих пар.

    Точность около 0.5 в начале и заметно выше после — иначе награда GAIL
    не несёт никакой информации.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # Эксперт: наблюдения из узкой области; политика: из широкой.
    expert = Demonstrations(
        rng.normal(loc=3.0, scale=0.1, size=(512, OBS_DIM)).astype(np.float32),
        np.zeros((512, 1), dtype=np.int64),
    )
    gail = GAIL(
        Discriminator(OBS_DIM, BRANCHES, hidden_sizes=(64, 64)),
        expert,
        GAILConfig(learning_rate=1e-3, batch_size=128, updates_per_batch=1,
                   gradient_penalty=0.0, label_smoothing=0.0),
        seed=0,
    )

    policy_obs = rng.normal(loc=-3.0, scale=0.1, size=(512, OBS_DIM)).astype(np.float32)
    policy_action = np.full((512, 1), 3, dtype=np.int64)

    stats = {}
    for _ in range(200):
        stats = gail.update((policy_obs, policy_action))

    assert stats["discriminator_accuracy"] > 0.9
    # И награда у эксперта обязана быть выше, чем у политики.
    assert gail.reward(expert.obs[:64], expert.action[:64]).mean() > \
           gail.reward(policy_obs[:64], policy_action[:64]).mean()


def test_gradient_penalty_is_finite_and_nonnegative():
    gail = make_gail(GAILConfig(batch_size=32, updates_per_batch=1, gradient_penalty=10.0))
    rng = np.random.default_rng(0)
    stats = gail.update((rng.normal(size=(64, OBS_DIM)).astype(np.float32),
                         rng.integers(0, 4, size=(64, 1)).astype(np.int64)))
    assert stats["gradient_penalty"] >= 0.0
    assert np.isfinite(stats["gradient_penalty"])


def test_gail_config_validates_weights():
    with pytest.raises(ValueError, match="env_reward_weight"):
        GAILConfig(env_reward_weight=1.5)
    with pytest.raises(ValueError, match="label_smoothing"):
        GAILConfig(label_smoothing=0.9)
