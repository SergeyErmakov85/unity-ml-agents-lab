"""Табличные методы: модель MDP, Value Iteration, Q-learning, экспорт таблицы."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from labrl.algos.tabular.q_learning import QLearning, QLearningConfig, Transition
from labrl.algos.tabular.value_iteration import (
    ValueIteration,
    ValueIterationConfig,
    optimality_rate,
    policy_agreement,
)
from labrl.envs.gridworld_mdp import GridWorldMDP, check_against_env
from labrl.nets.tabular import OneHotQTable

MDP = GridWorldMDP()


# --- модель MDP ---------------------------------------------------------


def test_state_indexing_roundtrips():
    for s in range(MDP.num_states):
        assert MDP.index(MDP.cell(s)) == s


def test_walls_block_movement():
    """Стена (1,2): из (1,1) ход на север не проходит."""
    s = MDP.index((1, 1))
    nxt, reward, terminated = MDP.step(s, 0)  # N
    assert nxt == s
    assert reward == pytest.approx(MDP.step_reward)
    assert not terminated


def test_boundary_blocks_movement():
    s = MDP.index((0, 0))
    assert MDP.step(s, 1)[0] == s  # S за границу
    assert MDP.step(s, 3)[0] == s  # W за границу


def test_goal_terminates_with_reward():
    s = MDP.index((4, 3))
    nxt, reward, terminated = MDP.step(s, 0)  # N -> (4,4) = цель
    assert nxt == MDP.index((4, 4))
    assert terminated
    assert reward == pytest.approx(MDP.step_reward + MDP.goal_reward)


def test_trap_terminates_with_penalty():
    s = MDP.index((0, 3))
    nxt, reward, terminated = MDP.step(s, 2)  # E -> (1,3) = ловушка
    assert nxt == MDP.index((1, 3))
    assert terminated
    assert reward == pytest.approx(MDP.step_reward + MDP.trap_reward)


def test_terminal_states_are_absorbing_in_tables():
    """Из терминала — петля с нулевой наградой, иначе V(терминал) уплывёт."""
    next_state, reward, terminated = MDP.transition_tables()
    goal = MDP.index(MDP.goal)
    assert np.all(next_state[goal] == goal)
    assert np.all(reward[goal] == 0.0)
    assert np.all(terminated[goal])


def test_check_against_env_reports_mismatch():
    good = [(MDP.index((0, 0)), 0, MDP.index((0, 1)), MDP.step_reward, False)]
    assert check_against_env(MDP, good) == []

    bad = [(MDP.index((0, 0)), 0, MDP.index((4, 4)), 99.0, True)]
    problems = check_against_env(MDP, bad)
    assert len(problems) == 3  # состояние, награда и признак завершения


# --- Value Iteration ----------------------------------------------------


def test_value_iteration_converges_and_reaches_goal():
    result = ValueIteration(ValueIterationConfig(gamma=0.99)).solve(MDP)

    assert result.iterations < 10_000
    assert result.deltas[-1] < 1e-10
    # Значение стартовой клетки положительно: цель достижима и выгодна.
    assert result.values[MDP.index(MDP.start)] > 0.0
    # Значения терминальных состояний равны нулю по построению.
    assert result.values[MDP.index(MDP.goal)] == pytest.approx(0.0)


def test_optimal_policy_takes_shortest_path_from_start():
    """Из (0,0) кратчайший путь идёт на север или на восток, но не в стену."""
    result = ValueIteration().solve(MDP)
    action = int(result.policy[MDP.index(MDP.start)])
    nxt, _, _ = MDP.step(MDP.index(MDP.start), action)
    assert nxt != MDP.index(MDP.start), "оптимальная политика не должна упираться в стену"


def test_optimal_policy_avoids_traps():
    """Ни из одного состояния оптимальная политика не шагает в ловушку."""
    result = ValueIteration().solve(MDP)
    for s in range(MDP.num_states):
        if MDP.is_terminal(s) or MDP.is_wall(MDP.cell(s)):
            continue
        nxt, _, _ = MDP.step(s, int(result.policy[s]))
        assert not MDP.is_trap(MDP.cell(nxt)), f"из состояния {s}{MDP.cell(s)} политика ведёт в ловушку"


def test_value_iteration_rejects_gamma_one():
    with pytest.raises(ValueError, match="gamma"):
        ValueIterationConfig(gamma=1.0)


# --- Q-learning ---------------------------------------------------------


def rollout_q_learning(episodes: int = 4000, seed: int = 0) -> QLearning:
    """Обучает Q-learning на модели MDP — без Unity, для проверки самого алгоритма."""
    rng = np.random.default_rng(seed)
    algo = QLearning(MDP.num_states, MDP.num_actions, QLearningConfig(gamma=0.99, learning_rate=0.2))

    free_states = [s for s in range(MDP.num_states) if not MDP.is_terminal(s) and not MDP.is_wall(MDP.cell(s))]

    for episode in range(episodes):
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.6))
        state = int(rng.choice(free_states))
        for _ in range(MDP.max_steps):
            action = int(algo.act(np.array([state]), epsilon, rng)[0])
            next_state, reward, terminated = MDP.step(state, action)
            algo.update(Transition(
                state=np.array([state]),
                action=np.array([action]),
                reward=np.array([reward]),
                next_state=np.array([next_state]),
                terminated=np.array([terminated]),
            ))
            state = next_state
            if terminated:
                break
    return algo


def free_state_mask() -> np.ndarray:
    """Состояния, в которых выбор действия вообще имеет смысл."""
    return np.array([
        not MDP.is_terminal(s) and not MDP.is_wall(MDP.cell(s))
        for s in range(MDP.num_states)
    ])


def test_q_learning_finds_the_optimal_policy():
    """Главная проверка метода: обучение из опыта сходится к оптимуму DP.

    Сравнение идёт **не** с одной конкретной оптимальной политикой, а с
    множеством оптимальных действий. В сетке с симметриями из клетки на север
    и на восток ведут пути одинаковой длины: обе политики оптимальны, и
    требовать совпадения argmax значило бы проверять, как разорвалась ничья.
    """
    optimal = ValueIteration(ValueIterationConfig(gamma=0.99)).solve(MDP)
    algo = rollout_q_learning()

    rate = optimality_rate(algo.greedy_policy(), optimal.q, free_state_mask())

    assert rate == 1.0, (
        f"Q-learning выбрал неоптимальное действие в {(1 - rate):.0%} состояний:\n"
        f"выучено:\n{MDP.render_policy(algo.greedy_policy())}\n"
        f"оптимум (один из):\n{MDP.render_policy(optimal.policy)}"
    )


def test_optimality_criterion_accepts_ties_and_rejects_errors():
    """Критерий обязан принимать ничьи и отвергать заведомо плохую политику."""
    optimal = ValueIteration().solve(MDP)
    mask = free_state_mask()

    assert optimality_rate(optimal.policy, optimal.q, mask) == 1.0

    always_south = np.ones(MDP.num_states, dtype=np.int64)
    assert optimality_rate(always_south, optimal.q, mask) < 0.5


def test_policy_agreement_is_stricter_than_optimality():
    """Диагностическая метрика: совпадение действий может быть < 1 при оптимальной политике."""
    optimal = ValueIteration().solve(MDP)
    algo = rollout_q_learning()
    mask = free_state_mask()

    assert policy_agreement(algo.greedy_policy(), optimal.policy, mask) <= 1.0
    assert optimality_rate(algo.greedy_policy(), optimal.q, mask) == 1.0


def test_q_learning_bootstraps_on_truncation_but_not_termination():
    """Разница terminated / truncated обязана менять цель обновления."""
    cfg = QLearningConfig(gamma=0.9, learning_rate=1.0, initial_q=0.0)

    terminated_algo = QLearning(2, 2, cfg)
    terminated_algo.q[1, :] = 5.0
    terminated_algo.update(Transition(
        state=np.array([0]), action=np.array([0]), reward=np.array([1.0]),
        next_state=np.array([1]), terminated=np.array([True]),
    ))
    assert terminated_algo.q[0, 0] == pytest.approx(1.0)  # будущее обнулено

    truncated_algo = QLearning(2, 2, cfg)
    truncated_algo.q[1, :] = 5.0
    truncated_algo.update(Transition(
        state=np.array([0]), action=np.array([0]), reward=np.array([1.0]),
        next_state=np.array([1]), terminated=np.array([False]),
        truncated=np.array([True]),
    ))
    assert truncated_algo.q[0, 0] == pytest.approx(1.0 + 0.9 * 5.0)  # бутстрэппинг выполнен


def test_duplicate_pairs_in_batch_are_all_applied():
    """Две арены дали один и тот же (s, a) — потерять одно из обновлений нельзя."""
    algo = QLearning(2, 2, QLearningConfig(gamma=0.0, learning_rate=0.5, initial_q=0.0))
    algo.update(Transition(
        state=np.array([0, 0]), action=np.array([0, 0]), reward=np.array([1.0, 1.0]),
        next_state=np.array([1, 1]), terminated=np.array([True, True]),
    ))
    # Одно обновление дало бы 0.5; два — 1.0.
    assert algo.q[0, 0] == pytest.approx(1.0)


def test_empty_batch_is_a_noop():
    algo = QLearning(5, 4)
    before = algo.q.copy()
    metrics = algo.update(Transition(
        state=np.array([], dtype=np.int64), action=np.array([], dtype=np.int64),
        reward=np.array([]), next_state=np.array([], dtype=np.int64),
        terminated=np.array([], dtype=bool),
    ))
    assert np.array_equal(algo.q, before)
    assert metrics["updates"] == 0.0


def test_epsilon_zero_is_purely_greedy():
    algo = QLearning(3, 4)
    algo.q[0] = np.array([0.0, 0.0, 9.0, 0.0])
    rng = np.random.default_rng(0)
    actions = algo.act(np.zeros(50, dtype=np.int64), epsilon=0.0, rng=rng)
    assert np.all(actions == 2)


def test_state_dict_roundtrip():
    algo = rollout_q_learning(episodes=50)
    restored = QLearning(MDP.num_states, MDP.num_actions)
    restored.load_state_dict(algo.state_dict())
    assert np.array_equal(restored.q, algo.q)


def test_load_state_dict_rejects_wrong_shape():
    algo = QLearning(25, 4)
    with pytest.raises(ValueError, match="форма таблицы"):
        algo.load_state_dict({"q": np.zeros((10, 4))})


# --- таблица как экспортируемый модуль ----------------------------------


def test_one_hot_table_reproduces_q_rows_exactly():
    """onehot(s) @ Qᵀ обязан дать ровно строку Q[s] — иначе в Unity уедет не та политика."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(25, 4)).astype(np.float32)
    module = OneHotQTable.from_table(q)

    obs = np.eye(25, dtype=np.float32)
    with torch.no_grad():
        out = module(torch.from_numpy(obs)).numpy()

    assert np.allclose(out, q, atol=1e-6)


def test_one_hot_table_roundtrips_to_table():
    q = np.random.default_rng(1).normal(size=(25, 4)).astype(np.float32)
    assert np.allclose(OneHotQTable.from_table(q).to_table(), q, atol=1e-6)


def test_policy_module_argmax_matches_greedy_policy():
    """Жадная политика таблицы и argmax экспортируемого модуля обязаны совпадать."""
    algo = rollout_q_learning(episodes=500)
    module = algo.policy_module()

    obs = np.eye(MDP.num_states, dtype=np.float32)
    with torch.no_grad():
        module_actions = module(torch.from_numpy(obs)).argmax(dim=1).numpy()

    assert np.array_equal(module_actions, algo.greedy_policy())


def test_from_table_rejects_non_2d():
    with pytest.raises(ValueError, match="двумерной"):
        OneHotQTable.from_table(np.zeros(25))
