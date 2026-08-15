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
