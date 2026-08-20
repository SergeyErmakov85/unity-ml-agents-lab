"""Учебный план: правило перевода и защиты от ложных срабатываний.

Ошибки учебного плана не роняют обучение — они делают его бессмысленным.
Слишком ранний перевод оставляет агента без сигнала на следующем уровне;
слишком поздний тратит бюджет впустую; а измерение по метрике, зависящей
от сложности, даёт план, который переводит не тогда, когда агент научился.
"""

from __future__ import annotations

import pytest

from labrl.train.curriculum import Curriculum, Lesson, build_curriculum

PLAN = [
    {"difficulty": 0.0, "threshold": 0.70, "min_steps": 1000},
    {"difficulty": 0.5, "threshold": 0.60, "min_steps": 2000},
    {"difficulty": 1.0, "threshold": 1.00},
]


def plan() -> Curriculum:
    return build_curriculum(PLAN)


# --- валидация ----------------------------------------------------------


def test_empty_plan_rejected():
    with pytest.raises(ValueError, match="хотя бы один урок"):
        Curriculum([])


def test_lessons_must_increase_in_difficulty():
    """План, идущий вниз по сложности, — почти наверняка опечатка в конфиге."""
    with pytest.raises(ValueError, match="по возрастанию"):
        Curriculum([Lesson(0.5, 0.7), Lesson(0.0, 0.7)])


def test_difficulty_out_of_range_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Lesson(difficulty=1.5, threshold=0.7)


# --- правило перевода ---------------------------------------------------


def test_starts_at_first_lesson():
    c = plan()
    assert c.index == 0
    assert c.difficulty == pytest.approx(0.0)
    assert not c.finished


def test_no_advance_below_threshold():
    c = plan()
    assert c.report(5000, 0.69) is False
    assert c.index == 0


def test_advance_at_threshold():
    c = plan()
    assert c.report(5000, 0.70) is True
    assert c.index == 1
    assert c.difficulty == pytest.approx(0.5)


def test_min_steps_blocks_early_advance():
    """Защита от перевода по одному удачному замеру.

    Доля успехов, посчитанная по двум-трём десяткам эпизодов, шумит:
    одно значение выше порога ничего не доказывает.
    """
    c = plan()
    assert c.report(500, 1.0) is False, "урок длится меньше min_steps"
    assert c.index == 0
    assert c.report(1000, 1.0) is True
    assert c.index == 1


def test_min_steps_counts_from_lesson_start_not_from_zero():
    """Второй урок обязан длиться свои min_steps, а не считаться от начала."""
    c = plan()
    c.report(1000, 1.0)              # переход на урок 2 (min_steps = 2000)
    assert c.report(2500, 1.0) is False, "с начала урока прошло 1500 < 2000"
    assert c.report(3000, 1.0) is True


def test_last_lesson_never_advances():
    """С последнего урока переходить некуда — сколько бы ни намерили."""
    c = plan()
    c.report(1000, 1.0)
    c.report(3000, 1.0)
    assert c.finished
    assert c.report(100_000, 1.0) is False
    assert c.difficulty == pytest.approx(1.0)


def test_plan_does_not_go_back():
    """План не откатывается на упавшей метрике — сознательное решение.

    Доля успехов падает сразу после КАЖДОГО повышения: это ожидаемое
    поведение, а не признак беды. Автоматический откат по такому сигналу
    даёт колебания между двумя уровнями и обучение, которое никуда не идёт.
    """
    c = plan()
    c.report(1000, 1.0)
    assert c.index == 1
    c.report(2000, 0.0)
    assert c.index == 1, "план обязан остаться на месте, а не откатиться"


def test_history_records_transitions():
    c = plan()
    c.report(1000, 0.9)
    c.report(4000, 0.8)
    assert [(step, index) for step, index, _ in c.history] == [(1000, 1), (4000, 2)]
    assert "урок 3 из 3" in c.describe()
