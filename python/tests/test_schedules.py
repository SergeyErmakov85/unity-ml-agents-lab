"""Расписания гиперпараметров."""

from __future__ import annotations

import pytest

from labrl.utils.schedules import (
    ConstantSchedule,
    ExponentialSchedule,
    LinearSchedule,
    build_schedule,
)


def test_constant():
    s = ConstantSchedule(0.1)
    assert s(0) == s(10_000) == pytest.approx(0.1)


def test_linear_endpoints_and_clamping():
    s = LinearSchedule(start=1.0, end=0.05, decay_steps=100)
    assert s(0) == pytest.approx(1.0)
    assert s(50) == pytest.approx(0.525)
    assert s(100) == pytest.approx(0.05)
    # После окончания спада значение не уходит ниже end — иначе ε стало бы
    # отрицательным и разведка выключилась бы совсем.
    assert s(10_000) == pytest.approx(0.05)
    assert s(-5) == pytest.approx(1.0)


def test_linear_rejects_nonpositive_decay():
    with pytest.raises(ValueError, match="decay_steps"):
        LinearSchedule(1.0, 0.0, 0)


def test_exponential_respects_floor():
    s = ExponentialSchedule(start=1.0, end=0.1, rate=0.5)
    assert s(0) == pytest.approx(1.0)
    assert s(1) == pytest.approx(0.5)
    assert s(100) == pytest.approx(0.1)


def test_exponential_rejects_invalid_rate():
    with pytest.raises(ValueError, match="rate"):
        ExponentialSchedule(1.0, 0.1, 1.5)


def test_build_schedule_from_config_block():
    s = build_schedule({"type": "linear", "start": 1.0, "end": 0.0, "decay_steps": 10})
    assert isinstance(s, LinearSchedule)
    assert s(10) == pytest.approx(0.0)


def test_build_schedule_rejects_unknown_type():
    with pytest.raises(ValueError, match="неизвестный тип"):
        build_schedule({"type": "cosine", "start": 1.0})


# --- число вместо блока -------------------------------------------------


def test_build_schedule_accepts_a_plain_number():
    """`docs/03_CONVENTIONS.md`, §6 разрешает и число, и блок.

    Без этой ветки конфиг с `epsilon: 0.0` падал с
    `TypeError: 'float' object is not subscriptable` — и не при разборе
    конфига, а уже после открытия среды Unity, то есть через минуту ожидания.
    """
    schedule = build_schedule(0.25)
    assert schedule(0) == pytest.approx(0.25)
    assert schedule(1_000_000) == pytest.approx(0.25)


def test_build_schedule_accepts_int():
    assert build_schedule(1)(42) == pytest.approx(1.0)


def test_build_schedule_rejects_nonsense():
    with pytest.raises(ValueError, match="числом или блоком"):
        build_schedule("linear")
