"""Многорукие бандиты: правила выбора и оценка ценности рук (`E00_Bandit`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from labrl.algos.bandits import Bandit, BanditBatch, BanditConfig


def make(strategy: str = "eps_greedy", **kwargs) -> Bandit:
    return Bandit(num_actions=4, cfg=BanditConfig(strategy=strategy, **kwargs))


# --- оценка ценности ----------------------------------------------------


def test_values_equal_sample_mean():
    """Инкрементальное обновление Q ← Q + (r − Q)/N даёт ровно выборочное среднее."""
    algo = make()
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0]
    for r in rewards:
        algo.update(BanditBatch(action=np.array([2]), reward=np.array([r])))

    assert algo.counts[2] == len(rewards)
    assert algo.values[2] == pytest.approx(np.mean(rewards))
    # Нетронутые руки остаются на начальном значении.
    assert algo.values[0] == pytest.approx(0.0)


def test_update_is_sequential_within_batch():
    """Два исхода одной руки в одном батче применяются последовательно.

    Наивная векторизация применила бы оба к одной и той же «старой» оценке,
    и среднее вышло бы неверным.
    """
    algo = make()
    algo.update(BanditBatch(action=np.array([1, 1]), reward=np.array([1.0, 0.0])))
    assert algo.counts[1] == 2
    assert algo.values[1] == pytest.approx(0.5)


def test_empty_batch_is_noop():
    algo = make()
    before = algo.values.copy()
    metrics = algo.update(BanditBatch(action=np.array([], dtype=np.int64), reward=np.array([])))
    assert np.array_equal(algo.values, before)
    assert metrics["value_error_abs"] == 0.0


# --- правила выбора -----------------------------------------------------


def test_eps_greedy_without_exploration_is_argmax():
    algo = make()
    algo.values[:] = [0.1, 0.9, 0.2, 0.3]
    rng = np.random.default_rng(0)
    assert np.all(algo.act(8, epsilon=0.0, rng=rng) == 1)


def test_eps_greedy_explores_at_epsilon_one():
    """При ε = 1 выбор равномерен: должны встретиться все руки."""
    algo = make()
    algo.values[:] = [0.0, 1.0, 0.0, 0.0]
    rng = np.random.default_rng(0)
    chosen = algo.act(400, epsilon=1.0, rng=rng)
    assert set(np.unique(chosen)) == {0, 1, 2, 3}


def test_ucb_tries_every_arm_before_repeating():
    """Ненажатая рука получает бесконечный приоритет — это и есть разведка UCB."""
    algo = make("ucb")
    rng = np.random.default_rng(0)
    seen = []
    for _ in range(algo.num_actions):
        arm = int(algo.act(1, epsilon=0.0, rng=rng)[0])
        seen.append(arm)
        algo.update(BanditBatch(action=np.array([arm]), reward=np.array([0.0])))
    assert sorted(seen) == [0, 1, 2, 3]


def test_ucb_bonus_shrinks_with_pulls():
    """Ширина доверительного интервала падает по мере накопления опыта."""
    algo = make("ucb")
    algo.update(BanditBatch(action=np.array([0, 1]), reward=np.array([0.5, 0.5])))
    early = algo.ucb_scores()[0] - algo.values[0]

    for _ in range(50):
        algo.update(BanditBatch(action=np.array([0]), reward=np.array([0.5])))
    late = algo.ucb_scores()[0] - algo.values[0]

    assert late < early


def test_thompson_prefers_arm_with_more_successes():
    """После явного перевеса апостериорное почти всегда выбирает лучшую руку."""
    algo = make("thompson")
    algo.update(BanditBatch(action=np.full(60, 3), reward=np.ones(60)))
    algo.update(BanditBatch(action=np.full(60, 0), reward=np.zeros(60)))

    rng = np.random.default_rng(0)
    chosen = algo.act(200, epsilon=0.0, rng=rng)
    assert (chosen == 3).mean() > 0.9


def test_thompson_explores_without_data():
    """Без данных апостериорные распределения одинаковы — выбор размазан."""
    algo = make("thompson")
    rng = np.random.default_rng(0)
    chosen = algo.act(400, epsilon=0.0, rng=rng)
    assert set(np.unique(chosen)) == {0, 1, 2, 3}


def test_greedy_action_ignores_exploration():
    """Итоговая политика жадная у всех стратегий — её и оценивают, её и экспортируют."""
    for strategy in ("eps_greedy", "ucb", "thompson"):
        algo = make(strategy)
        algo.values[:] = [0.1, 0.2, 0.7, 0.3]
        assert np.all(algo.greedy_action(5) == 2)


# --- экспорт и сохранение -----------------------------------------------


def test_policy_module_reproduces_values_on_constant_observation():
    """Наблюдение среды — константа 1.0, и слой обязан выдать ровно оценки рук."""
    algo = make()
    algo.values[:] = [0.1, 0.9, 0.2, 0.3]
    module = algo.policy_module()

    obs = torch.ones(7, 1)
    out = module(obs).detach().numpy()

    assert out.shape == (7, 4)
    np.testing.assert_allclose(out[0], algo.values, rtol=0, atol=1e-6)
    assert np.all(out.argmax(axis=1) == algo.greedy_action(7))


def test_state_dict_round_trip():
    algo = make("thompson")
    algo.update(BanditBatch(action=np.array([0, 1, 2]), reward=np.array([1.0, 0.0, 1.0])))

    restored = make("thompson")
    restored.load_state_dict(algo.state_dict())

    np.testing.assert_allclose(restored.values, algo.values)
    np.testing.assert_array_equal(restored.counts, algo.counts)
    np.testing.assert_array_equal(restored.successes, algo.successes)
    assert restored.pulls == algo.pulls


def test_regret_per_pull_counts_only_suboptimal_choices():
    algo = make()
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])
    algo.update(BanditBatch(action=np.array([3, 3, 1]), reward=np.array([1.0, 0.0, 1.0])))
    # Две оптимальные руки дают нулевое сожаление, одна рука 1 — сожаление 0.4.
    assert algo.regret_per_pull(probabilities) == pytest.approx(0.4 / 3)


# --- валидация конфигурации ---------------------------------------------


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="strategy"):
        BanditConfig(strategy="softmax")


def test_single_arm_rejected():
    with pytest.raises(ValueError, match="рук"):
        Bandit(num_actions=1)
