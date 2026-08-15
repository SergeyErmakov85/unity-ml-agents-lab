"""Обобщённое преимущество (GAE) и различение `terminated` / `truncated`.

Тот же класс ошибок, что покрыт для DQN в `test_vec_env_termination.py`,
но для on-policy методов. Если обрыв по времени обнулит будущее, A2C и PPO
выучат, что нехватка времени равносильна провалу; обучение при этом не падает,
оно тихо портится. Требование 8.3 предписывает покрыть это тестом.
"""

from __future__ import annotations

import numpy as np
import pytest

from labrl.buffers.rollout import RolloutBuffer

GAMMA = 0.9
LAM = 0.8


def make_buffer(num_envs: int = 1) -> RolloutBuffer:
    return RolloutBuffer(num_envs=num_envs, gamma=GAMMA, gae_lambda=LAM)


def add(buffer: RolloutBuffer, slot: int, reward: float, value: float,
        terminated: bool = False, truncated: bool = False, bootstrap: float = 0.0) -> None:
    buffer.add(
        slot=slot,
        obs=np.zeros(2, dtype=np.float32),
        action=np.zeros(1, dtype=np.float32),
        log_prob=0.0,
        value=value,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        bootstrap_value=bootstrap,
    )


# --- базовая свёртка ----------------------------------------------------


def test_gae_matches_manual_computation_without_episode_end():
    """Два шага без завершения: значения совпадают с формулой из документации."""
    buffer = make_buffer()
    add(buffer, 0, reward=1.0, value=0.5)
    add(buffer, 0, reward=2.0, value=1.0)

    batch = buffer.compute(last_values=np.array([3.0]))

    delta1 = 2.0 + GAMMA * 3.0 - 1.0
    delta0 = 1.0 + GAMMA * 1.0 - 0.5
    expected = np.array([delta0 + GAMMA * LAM * delta1, delta1])

    np.testing.assert_allclose(batch.advantage, expected, rtol=1e-6)


def test_returns_equal_advantage_plus_value():
    """Цель критика — то же TD(λ): отдельная сумма наград не нужна."""
    buffer = make_buffer()
    add(buffer, 0, reward=1.0, value=0.5)
    add(buffer, 0, reward=2.0, value=1.0)

    batch = buffer.compute(last_values=np.array([3.0]))
    np.testing.assert_allclose(batch.returns, batch.advantage + batch.value, rtol=1e-6)


def test_lambda_zero_leaves_one_step_td_error():
    """При λ = 0 преимущество вырождается в одношаговую TD-ошибку."""
    buffer = RolloutBuffer(num_envs=1, gamma=GAMMA, gae_lambda=0.0)
    add(buffer, 0, reward=1.0, value=0.5)
    add(buffer, 0, reward=2.0, value=1.0)

    batch = buffer.compute(last_values=np.array([3.0]))
    np.testing.assert_allclose(
        batch.advantage,
        np.array([1.0 + GAMMA * 1.0 - 0.5, 2.0 + GAMMA * 3.0 - 1.0]),
        rtol=1e-6,
    )


# --- завершение против обрыва -------------------------------------------


def test_terminated_zeroes_the_future():
    """Истинное завершение: будущего нет, бутстрэппинг не выполняется."""
    buffer = make_buffer()
    add(buffer, 0, reward=1.0, value=0.4, terminated=True, bootstrap=99.0)

    batch = buffer.compute(last_values=np.array([50.0]))
    # Ни bootstrap, ни last_values не должны попасть в расчёт.
    assert batch.advantage[0] == pytest.approx(1.0 - 0.4)


def test_truncated_bootstraps_from_critic():
    """Обрыв по MaxStep: эпизод продолжался бы, ценность берётся у критика."""
    buffer = make_buffer()
    add(buffer, 0, reward=1.0, value=0.4, truncated=True, bootstrap=2.0)

    batch = buffer.compute(last_values=np.array([50.0]))
    assert batch.advantage[0] == pytest.approx(1.0 + GAMMA * 2.0 - 0.4)


def test_terminated_and_truncated_differ():
    """Прямое сравнение: путать эти случаи нельзя, разница видна в числах."""
    terminated = make_buffer()
    add(terminated, 0, reward=0.0, value=0.0, terminated=True, bootstrap=5.0)

    truncated = make_buffer()
    add(truncated, 0, reward=0.0, value=0.0, truncated=True, bootstrap=5.0)

    a = terminated.compute(last_values=np.zeros(1)).advantage[0]
    b = truncated.compute(last_values=np.zeros(1)).advantage[0]
    assert a == pytest.approx(0.0)
    assert b == pytest.approx(GAMMA * 5.0)


def test_chain_resets_across_episode_boundary():
    """Преимущество шага не должно перетекать через конец эпизода.

    Шаг 0 завершает эпизод; шаг 1 — начало следующего. Значение шага 0 обязано
    зависеть только от собственной награды, каким бы большим ни было
    преимущество шага 1.
    """
    buffer = make_buffer()
    add(buffer, 0, reward=1.0, value=0.0, terminated=True)
    add(buffer, 0, reward=100.0, value=0.0)

    batch = buffer.compute(last_values=np.array([0.0]))
    assert batch.advantage[0] == pytest.approx(1.0)
    assert batch.advantage[1] == pytest.approx(100.0)


# --- несколько слотов ---------------------------------------------------


def test_slots_are_independent():
    """Слоты — независимые последовательности; асинхронность их не смешивает."""
    buffer = make_buffer(num_envs=3)
    add(buffer, 0, reward=1.0, value=0.0)
    add(buffer, 2, reward=1.0, value=0.0, terminated=True)
    # Слот 1 не дал ни одного перехода — он просто отсутствует в батче.

    batch = buffer.compute(last_values=np.array([10.0, 999.0, 999.0]))

    assert len(batch) == 2
    np.testing.assert_allclose(batch.advantage, np.array([1.0 + GAMMA * 10.0, 1.0]), rtol=1e-6)


def test_len_counts_all_slots():
    buffer = make_buffer(num_envs=2)
    add(buffer, 0, reward=0.0, value=0.0)
    add(buffer, 1, reward=0.0, value=0.0)
    assert len(buffer) == 2
    buffer.clear()
    assert len(buffer) == 0


def test_empty_rollout_is_an_error():
    """Пустой роллаут — ошибка сбора, а не повод молча обновиться на нуле."""
    with pytest.raises(RuntimeError, match="пуст"):
        make_buffer().compute(last_values=np.zeros(1))


def test_last_values_shape_is_checked():
    buffer = make_buffer(num_envs=2)
    add(buffer, 0, reward=0.0, value=0.0)
    with pytest.raises(ValueError, match="last_values"):
        buffer.compute(last_values=np.zeros(3))
