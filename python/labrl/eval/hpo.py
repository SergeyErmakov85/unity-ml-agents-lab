"""Подбор гиперпараметров: случайный поиск и последовательное деление пополам.

Зачем это здесь и почему не Optuna
-----------------------------------
Урок 3.6 курса разбирает HPO через Optuna и W&B Sweeps. Здесь те же два
базовых алгоритма написаны своим кодом — по той же причине, по которой
своими руками написаны PPO и SAC: **учебный код не должен прятать
происходящее за библиотечным вызовом**. Оба алгоритма умещаются
в полторы сотни строк, и в них нечего скрывать.

Запрет 16.5 на сторонние библиотеки касается реализаций **алгоритмов
обучения**, и Optuna под него не подпадает: она не RL-библиотека. Подключить
её к тому же интерфейсу :class:`Trial` пользователь может сам — это
десяток строк.

Два алгоритма и в чём между ними разница
-----------------------------------------
**Случайный поиск** (Bergstra & Bengio, 2012). Берёт ``n`` случайных точек
пространства и обучает каждую полным бюджетом. Прост, честен и незаменим
как точка отсчёта: любой более умный метод обязан его обыгрывать, иначе
он не нужен.

Почему **случайный**, а не полный перебор по сетке. При ``k`` параметрах
сетка из ``m`` значений требует ``m^k`` прогонов, и при этом каждый параметр
принимает всего ``m`` различных значений. Случайный поиск за те же ``n``
прогонов даёт ``n`` различных значений **каждому** параметру. Поскольку
на практике важны один-два параметра из десяти, а какие именно — заранее
неизвестно, случайный поиск систематически выигрывает.

**Последовательное деление пополам** (successive halving, Jamieson &
Talwalkar, 2016). Обучает все точки коротким бюджетом, отбрасывает худшую
половину, оставшимся удваивает бюджет, и так далее. За тот же суммарный
бюджет проверяет **втрое-вчетверо больше** конфигураций.

Его известная слабость: конфигурация, медленно стартующая, но сильная
в пределе, будет отброшена рано. Это не дефект реализации, а свойство
метода — и повод не доверять ему одному.

Что здесь НЕ реализовано
------------------------
Байесовская оптимизация и TPE (тоже из урока 3.6). Они требуют суррогатной
модели пространства и правила выбора следующей точки — это отдельная работа,
и без неё две базовые схемы уже дают учащемуся рабочий инструмент.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

#: Функция обучения: получает точку пространства и бюджет шагов,
#: возвращает измеренное качество (чем больше, тем лучше).
Objective = Callable[[Mapping[str, Any], int], float]


@dataclass
class SearchSpace:
    """Пространство поиска: по описанию на каждый гиперпараметр.

    Поддерживаются три вида:

    ``("uniform", low, high)``
        равномерно на отрезке — для величин, у которых важна разность
        (например `clip_range`);
    ``("log_uniform", low, high)``
        равномерно по логарифму — для величин, у которых важно **отношение**.
        Скорость обучения 1e-4 и 1e-3 отличаются вдесятеро, а 1e-3 и 2e-3 —
        вдвое: равномерная выборка на отрезке отдала бы почти все точки
        большим значениям;
    ``("choice", [a, b, c])``
        из списка — для дискретных величин (размер батча, число эпох).

    Пример::

        space = SearchSpace({
            "learning_rate": ("log_uniform", 1e-5, 1e-3),
            "clip_range":    ("uniform", 0.1, 0.4),
            "epochs":        ("choice", [2, 4, 8]),
        })
    """

    spec: Mapping[str, tuple]

    def __post_init__(self) -> None:
        for name, description in self.spec.items():
            kind = description[0]
            if kind not in ("uniform", "log_uniform", "choice"):
                raise ValueError(
                    f"{name}: неизвестный вид {kind!r}; "
                    "поддерживаются uniform, log_uniform, choice"
                )
            if kind in ("uniform", "log_uniform"):
                low, high = description[1], description[2]
                if not low < high:
                    raise ValueError(f"{name}: нижняя граница {low} не меньше верхней {high}")
                if kind == "log_uniform" and low <= 0:
                    raise ValueError(
                        f"{name}: log_uniform требует положительных границ, получено {low}"
                    )
            elif not description[1]:
                raise ValueError(f"{name}: пустой список вариантов")

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Случайная точка пространства."""
        point: dict[str, Any] = {}
        for name, description in self.spec.items():
            kind = description[0]
            if kind == "uniform":
                point[name] = float(rng.uniform(description[1], description[2]))
            elif kind == "log_uniform":
                low, high = math.log(description[1]), math.log(description[2])
                point[name] = float(math.exp(rng.uniform(low, high)))
            else:
                options = list(description[1])
                point[name] = options[int(rng.integers(len(options)))]
        return point

    def describe(self) -> str:
        lines = [f"параметров: {len(self.spec)}"]
        for name, description in self.spec.items():
            if description[0] == "choice":
                lines.append(f"  {name:<20} choice{tuple(description[1])}")
            else:
                lines.append(f"  {name:<20} {description[0]}({description[1]}, {description[2]})")
        return "\n".join(lines)


@dataclass
class Trial:
    """Одна испытанная конфигурация.

    Attributes:
        params: точка пространства.
        score: измеренное качество; больше — лучше.
        budget: бюджет шагов, на котором получено значение.
        wall_time: время прогона, с.
    """

    params: dict[str, Any]
    score: float
    budget: int
    wall_time: float = 0.0

    def describe(self) -> str:
        body = ", ".join(
            f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.params.items()
        )
        return f"score={self.score:+.4f} @ {self.budget:>7} шагов | {body}"


@dataclass
class SearchResult:
    """Итог подбора."""

    trials: list[Trial] = field(default_factory=list)
    #: Сколько шагов среды суммарно потрачено — главная цена метода.
    total_budget: int = 0
    wall_time: float = 0.0

    @property
    def best(self) -> Trial:
        if not self.trials:
            raise RuntimeError("ни одной завершённой конфигурации")
        # При равных значениях предпочитается та, что измерена на большем
        # бюджете: её оценка надёжнее.
        return max(self.trials, key=lambda t: (t.score, t.budget))

    def top(self, count: int = 5) -> list[Trial]:
        return sorted(self.trials, key=lambda t: (-t.score, -t.budget))[:count]

    def describe(self, count: int = 5) -> str:
        lines = [
            f"конфигураций испытано: {len(self.trials)}, "
            f"суммарный бюджет: {self.total_budget} шагов, время: {self.wall_time:.1f} с",
            f"лучшая: {self.best.describe()}",
            f"первые {count}:",
        ]
        lines += [f"  {i + 1}. {t.describe()}" for i, t in enumerate(self.top(count))]
        return "\n".join(lines)


def random_search(
    space: SearchSpace,
    objective: Objective,
    trials: int,
    budget: int,
    seed: int = 0,
    on_trial: Callable[[int, Trial], None] | None = None,
) -> SearchResult:
    """Случайный поиск: ``trials`` точек, каждая обучается бюджетом ``budget``.

    Точка отсчёта для любого более умного метода. Если делением пополам
    не удаётся обыграть случайный поиск при равном суммарном бюджете,
    деление не нужно.
    """
    if trials <= 0:
        raise ValueError(f"trials должно быть > 0, получено {trials}")

    rng = np.random.default_rng(seed)
    result = SearchResult()
    started = time.perf_counter()

    for index in range(trials):
        params = space.sample(rng)
        trial_started = time.perf_counter()
        score = float(objective(params, budget))
        trial = Trial(params=params, score=score, budget=budget,
                      wall_time=time.perf_counter() - trial_started)
        result.trials.append(trial)
        result.total_budget += budget
        if on_trial is not None:
            on_trial(index, trial)

    result.wall_time = time.perf_counter() - started
    return result


def successive_halving(
    space: SearchSpace,
    objective: Objective,
    trials: int,
    min_budget: int,
    reduction: int = 2,
    seed: int = 0,
    on_trial: Callable[[int, Trial], None] | None = None,
) -> SearchResult:
    """Последовательное деление пополам.

    Args:
        space: пространство поиска.
        objective: функция обучения.
        trials: сколько конфигураций взять на первом этапе.
        min_budget: бюджет первого этапа, шагов.
        reduction: во сколько раз сокращается число выживших и во столько же
            растёт бюджет. 2 — классическое «пополам»; 3 даёт более
            агрессивный отбор.
        seed: сид выборки точек.
        on_trial: колбэк ``(номер этапа, испытание)``.

    Returns:
        :class:`SearchResult`, где у каждой конфигурации записан **последний**
        бюджет, на котором она измерялась.

    Схема::

        этап 0:  n конфигураций × min_budget
        этап 1:  n/r           × min_budget·r
        этап 2:  n/r²          × min_budget·r²
        …пока не останется одна

    Суммарный бюджет каждого этапа примерно одинаков — в этом весь приём:
    ресурс перераспределяется от заведомо слабых конфигураций к тем,
    что подают надежды.

    Слабость метода: конфигурация, медленно стартующая, но сильная
    в пределе, будет отброшена рано. Это свойство алгоритма, а не дефект
    реализации, и повод не доверять ему одному.
    """
    if trials <= 0:
        raise ValueError(f"trials должно быть > 0, получено {trials}")
    if reduction < 2:
        raise ValueError(f"reduction должен быть >= 2, получено {reduction}")

    rng = np.random.default_rng(seed)
    result = SearchResult()
    started = time.perf_counter()

    survivors = [space.sample(rng) for _ in range(trials)]
    budget = int(min_budget)
    stage = 0
    # Итог по каждой конфигурации: ключ — её порядковый номер на старте.
    latest: dict[int, Trial] = {}
    indices = list(range(trials))

    while survivors:
        scored: list[tuple[int, Trial]] = []
        for slot, params in zip(indices, survivors):
            trial_started = time.perf_counter()
            score = float(objective(params, budget))
            trial = Trial(params=params, score=score, budget=budget,
                          wall_time=time.perf_counter() - trial_started)
            scored.append((slot, trial))
            latest[slot] = trial
            result.total_budget += budget
            if on_trial is not None:
                on_trial(stage, trial)

        if len(scored) == 1:
            break

        # Выживает лучшая доля 1/reduction, но не меньше одной конфигурации.
        keep = max(1, len(scored) // reduction)
        scored.sort(key=lambda pair: -pair[1].score)
        survivors = [trial.params for _, trial in scored[:keep]]
        indices = [slot for slot, _ in scored[:keep]]
        budget *= reduction
        stage += 1

    result.trials = [latest[slot] for slot in sorted(latest)]
    result.wall_time = time.perf_counter() - started
    return result
