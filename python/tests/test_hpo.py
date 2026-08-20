"""Подбор гиперпараметров: пространство поиска и две схемы отбора.

Ошибки HPO обходятся дорого именно потому, что незаметны: подбор
«отработал», выдал конфигурацию, и понять, что он искал не там,
можно только по отсутствию улучшения.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from labrl.eval.hpo import SearchSpace, random_search, successive_halving


def quadratic(params, budget):  # noqa: ANN001, ANN201
    """Игрушечная цель с известным максимумом в x = 0.5.

    Бюджет добавляет «точность измерения»: на малом бюджете значение
    зашумлено, на большом — точное. Это грубая модель того, как ведёт себя
    настоящее обучение.
    """
    exact = -((params["x"] - 0.5) ** 2)
    noise = 0.0 if budget >= 1000 else 0.05 * math.sin(budget * params["x"] * 97.0)
    return exact + noise


# --- пространство поиска ------------------------------------------------


def test_uniform_stays_within_bounds():
    space = SearchSpace({"x": ("uniform", -2.0, 3.0)})
    rng = np.random.default_rng(0)
    values = [space.sample(rng)["x"] for _ in range(500)]
    assert all(-2.0 <= v <= 3.0 for v in values)


def test_log_uniform_is_uniform_in_the_logarithm():
    """Половина точек обязана лежать ниже геометрического среднего.

    Именно ради этого свойства log_uniform и существует: скорость обучения
    1e-5 и 1e-4 отличаются вдесятеро, а 1e-3 и 2e-3 — вдвое, и равномерная
    выборка на отрезке отдала бы почти все точки большим значениям.
    """
    low, high = 1e-5, 1e-1
    space = SearchSpace({"lr": ("log_uniform", low, high)})
    rng = np.random.default_rng(0)
    values = np.array([space.sample(rng)["lr"] for _ in range(4000)])

    geometric_mean = math.sqrt(low * high)
    share = float((values < geometric_mean).mean())
    assert 0.45 < share < 0.55, f"доля точек ниже геометрического среднего {share:.2f}"


def test_choice_returns_only_listed_options():
    space = SearchSpace({"epochs": ("choice", [2, 4, 8])})
    rng = np.random.default_rng(0)
    values = {space.sample(rng)["epochs"] for _ in range(200)}
    assert values <= {2, 4, 8}


def test_space_rejects_bad_specifications():
    with pytest.raises(ValueError, match="неизвестный вид"):
        SearchSpace({"x": ("normal", 0, 1)})
    with pytest.raises(ValueError, match="не меньше верхней"):
        SearchSpace({"x": ("uniform", 3.0, 1.0)})
    with pytest.raises(ValueError, match="положительных границ"):
        SearchSpace({"x": ("log_uniform", -1.0, 1.0)})
    with pytest.raises(ValueError, match="пустой список"):
        SearchSpace({"x": ("choice", [])})


def test_sampling_is_reproducible():
    """Два запуска с одним сидом обязаны дать одни точки: иначе подбор
    невоспроизводим, и сравнивать его прогоны нельзя."""
    space = SearchSpace({"x": ("uniform", 0.0, 1.0), "e": ("choice", [1, 2, 3])})
    a = [space.sample(np.random.default_rng(7)) for _ in range(3)]
    b = [space.sample(np.random.default_rng(7)) for _ in range(3)]
    assert a == b


# --- случайный поиск ----------------------------------------------------


def test_random_search_finds_the_optimum_region():
    space = SearchSpace({"x": ("uniform", 0.0, 1.0)})
    result = random_search(space, quadratic, trials=60, budget=1000, seed=0)

    assert len(result.trials) == 60
    assert result.total_budget == 60 * 1000
    assert abs(result.best.params["x"] - 0.5) < 0.05


def test_random_search_reports_budget_honestly():
    """Суммарный бюджет — главная цена метода, и он обязан быть точным."""
    space = SearchSpace({"x": ("uniform", 0.0, 1.0)})
    result = random_search(space, quadratic, trials=7, budget=250, seed=0)
    assert result.total_budget == 7 * 250


def test_random_search_rejects_zero_trials():
    with pytest.raises(ValueError, match="trials"):
        random_search(SearchSpace({"x": ("uniform", 0.0, 1.0)}), quadratic, 0, 100)


def test_on_trial_callback_sees_every_trial():
    seen = []
    space = SearchSpace({"x": ("uniform", 0.0, 1.0)})
    random_search(space, quadratic, trials=5, budget=100, seed=0,
                  on_trial=lambda i, t: seen.append(i))
    assert seen == [0, 1, 2, 3, 4]


# --- деление пополам ----------------------------------------------------


def test_successive_halving_spends_less_than_full_random_search():
    """Главное обещание метода: за меньший бюджет — столько же конфигураций.

    16 конфигураций полным бюджетом 800 стоили бы 12 800 шагов; деление
    пополам с 16 конфигураций от 100 шагов тратит существенно меньше.
    """
    space = SearchSpace({"x": ("uniform", 0.0, 1.0)})
    halving = successive_halving(space, quadratic, trials=16, min_budget=100, seed=0)
    assert halving.total_budget < 16 * 800
    assert len(halving.trials) == 16


def test_successive_halving_gives_the_winner_the_largest_budget():
    """Смысл метода — перераспределить ресурс: выживший обязан быть измерен
    на самом большом бюджете."""
    space = SearchSpace({"x": ("uniform", 0.0, 1.0)})
    result = successive_halving(space, quadratic, trials=8, min_budget=100, seed=0)
    assert result.best.budget == max(t.budget for t in result.trials)


def test_successive_halving_finds_the_optimum_region():
    space = SearchSpace({"x": ("uniform", 0.0, 1.0)})
    result = successive_halving(space, quadratic, trials=32, min_budget=200, seed=0)
    assert abs(result.best.params["x"] - 0.5) < 0.1


def test_successive_halving_stage_count():
    """При 8 конфигурациях и reduction=2 этапов ровно четыре: 8 → 4 → 2 → 1."""
    stages = []
    space = SearchSpace({"x": ("uniform", 0.0, 1.0)})
    successive_halving(space, quadratic, trials=8, min_budget=100, seed=0,
                       on_trial=lambda stage, trial: stages.append(stage))
    assert max(stages) == 3
    # На каждом этапе — вдвое меньше конфигураций.
    from collections import Counter
    assert [Counter(stages)[s] for s in range(4)] == [8, 4, 2, 1]


def test_reduction_must_be_at_least_two():
    with pytest.raises(ValueError, match="reduction"):
        successive_halving(SearchSpace({"x": ("uniform", 0.0, 1.0)}),
                           quadratic, trials=4, min_budget=10, reduction=1)


def test_result_describe_is_readable():
    space = SearchSpace({"x": ("uniform", 0.0, 1.0)})
    result = random_search(space, quadratic, trials=3, budget=100, seed=0)
    text = result.describe(count=2)
    assert "конфигураций испытано: 3" in text
    assert "лучшая:" in text
