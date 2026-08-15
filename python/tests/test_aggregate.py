"""Агрегация по сидам: IQM и доверительный интервал (требование 12.3)."""

from __future__ import annotations

import numpy as np
import pytest

from labrl.eval.aggregate import interquartile_mean, stratified_bootstrap_iqm


def test_iqm_of_uniform_values_equals_that_value():
    assert interquartile_mean(np.full(8, 3.0)) == pytest.approx(3.0)


def test_iqm_drops_quarter_from_each_tail():
    values = np.arange(1.0, 9.0)  # 1..8: отбрасываются 1,2 и 7,8
    assert interquartile_mean(values) == pytest.approx(np.mean([3, 4, 5, 6]))


def test_iqm_is_robust_to_a_single_catastrophic_seed():
    """Ради этого свойства IQM и выбран вместо среднего.

    Один провалившийся сид из восьми сдвигает среднее почти на 12 %,
    а IQM — не меняет вовсе.
    """
    good = np.full(8, 1.0)
    with_outlier = good.copy()
    with_outlier[0] = -10.0

    assert np.mean(with_outlier) < 0.9 * np.mean(good)
    assert interquartile_mean(with_outlier) == pytest.approx(interquartile_mean(good))


def test_iqm_rejects_empty_input():
    with pytest.raises(ValueError, match="пустой массив"):
        interquartile_mean(np.array([]))


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    scores = rng.normal(loc=0.8, scale=0.05, size=(5, 20))

    result = stratified_bootstrap_iqm(scores, resamples=500, seed=0)

    assert result.ci_low <= result.iqm <= result.ci_high
    assert result.num_seeds == 5
    assert result.num_samples == 100
    assert 0.7 < result.iqm < 0.9


def test_bootstrap_is_reproducible_for_same_seed():
    scores = np.random.default_rng(1).normal(size=(4, 10))
    a = stratified_bootstrap_iqm(scores, resamples=200, seed=7)
    b = stratified_bootstrap_iqm(scores, resamples=200, seed=7)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_wider_spread_between_seeds_gives_wider_interval():
    """Стратификация по сидам обязана «увидеть» межсидовый разброс."""
    tight = np.tile(np.array([[0.80], [0.81], [0.79], [0.80]]), (1, 20))
    loose = np.tile(np.array([[0.2], [0.9], [0.5], [1.4]]), (1, 20))

    tight_ci = stratified_bootstrap_iqm(tight, resamples=500, seed=0)
    loose_ci = stratified_bootstrap_iqm(loose, resamples=500, seed=0)

    assert (loose_ci.ci_high - loose_ci.ci_low) > (tight_ci.ci_high - tight_ci.ci_low)


def test_single_seed_rejected():
    """Вывод по одному прогону запрещён инструкцией 12.3."""
    with pytest.raises(ValueError, match="минимум 2 сида"):
        stratified_bootstrap_iqm(np.array([[1.0, 2.0, 3.0]]))
