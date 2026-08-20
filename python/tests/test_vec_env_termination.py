"""Различение terminated и truncated — обязательный тест по требованию 8.3.

Почему этот тест существует. При обрыве по ``MaxStep`` ценность будущего
не равна нулю, и бутстрэппинг обязан выполняться:

    y = r + γ·V(s')   если truncated
    y = r             если terminated

Если обёртка сваливает обе причины в один флаг ``done``, обучение не падает —
оно тихо портится: агент учится считать «время вышло» катастрофой. Ошибку
такого рода в готовом прогоне не видно, поэтому она ловится здесь.
"""

from __future__ import annotations

import numpy as np
import pytest

from labrl.envs.unity_env import UnityEnvHandle
from labrl.envs.vec_unity_env import VecUnityEnv
from tests.fake_unity import FakeChannels, FakeUnityEnv, ScriptedStep, make_spec

OBS_DIM = 3
BEHAVIOR = "E99_Fake"


def make_vec(script: list[ScriptedStep], num_envs: int | None = None) -> VecUnityEnv:
    spec = make_spec(OBS_DIM)
    env = FakeUnityEnv(BEHAVIOR, OBS_DIM, spec, script)
    handle = UnityEnvHandle(env=env, channels=FakeChannels(), behavior_name=BEHAVIOR, spec=spec)
    return VecUnityEnv(handle, num_envs=num_envs)


def noop_actions(n: int) -> np.ndarray:
    return np.zeros((n, 1), dtype=np.int32)


def test_maxstep_interruption_is_truncated_not_terminated():
    """interrupted=True — это обрыв по времени, а не терминальное состояние."""
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1, 11: 0.2}),
        ScriptedStep(decisions={11: 0.4}, terminals={10: (0.9, 1.0, True)}),
    ])
    vec.reset()
    result = vec.step(noop_actions(2))

    assert result.truncated[0] and not result.terminated[0]
    assert not result.truncated[1] and not result.terminated[1]


def test_true_termination_is_terminated_not_truncated():
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1, 11: 0.2}),
        ScriptedStep(decisions={11: 0.4}, terminals={10: (0.9, 1.0, False)}),
    ])
    vec.reset()
    result = vec.step(noop_actions(2))

    assert result.terminated[0] and not result.truncated[0]


def test_both_reasons_in_one_step_are_kept_apart():
    """Два слота кончают эпизод по разным причинам одновременно."""
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1, 11: 0.2}),
        ScriptedStep(terminals={10: (0.9, 1.0, False), 11: (0.8, -1.0, True)}),
    ])
    vec.reset()
    result = vec.step(noop_actions(2))

    assert result.terminated.tolist() == [True, False]
    assert result.truncated.tolist() == [False, True]
    assert result.done.tolist() == [True, True]


def test_terminal_observation_survives_immediate_reset():
    """Начало нового эпизода не должно затирать терминальное наблюдение.

    Если арена стартует новый эпизод в том же шаге, `obs` содержит начальное
    наблюдение нового эпизода, а терминальное остаётся доступным в `final_obs`.
    Перепутать их — значит считать цель обновления по состоянию из другого
    эпизода.
    """
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1}),
        # Агент 10 завершил эпизод, агент 20 (новый эпизод той же арены)
        # сразу запросил решение.
        ScriptedStep(decisions={20: 0.5}, terminals={10: (0.9, 1.0, False)}),
    ])
    vec.reset()
    result = vec.step(noop_actions(1))

    assert result.terminated[0]
    assert result.final_obs[0][0] == pytest.approx([0.9] * OBS_DIM)
    assert result.obs[0][0] == pytest.approx([0.5] * OBS_DIM)
    assert result.active[0]


def test_slot_is_inactive_until_new_episode_requests_decision():
    """Слот без решения нельзя считать пригодным для действия."""
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1, 11: 0.2}),
        ScriptedStep(decisions={11: 0.3}, terminals={10: (0.9, 1.0, False)}),
        ScriptedStep(decisions={11: 0.4, 20: 0.7}),
    ])
    vec.reset()

    after_terminal = vec.step(noop_actions(2))
    assert after_terminal.active.tolist() == [False, True]

    after_respawn = vec.step(noop_actions(2))
    assert after_respawn.active.tolist() == [True, True]
    assert after_respawn.obs[0][0] == pytest.approx([0.7] * OBS_DIM)


def test_new_episode_reuses_the_same_slot():
    """Идентификатор агента меняется каждый эпизод — слот обязан остаться прежним."""
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1, 11: 0.2}),
        ScriptedStep(decisions={11: 0.3}, terminals={10: (0.9, 1.0, False)}),
        ScriptedStep(decisions={11: 0.4, 20: 0.7}),
        ScriptedStep(decisions={11: 0.5, 20: 0.8}),
    ])
    vec.reset()
    vec.step(noop_actions(2))
    vec.step(noop_actions(2))
    result = vec.step(noop_actions(2))

    # Слот 0 всё время принадлежит первой арене, хотя агент в нём уже другой.
    assert result.obs[0][0] == pytest.approx([0.8] * OBS_DIM)
    assert result.obs[0][1] == pytest.approx([0.5] * OBS_DIM)


def test_reward_of_terminal_step_is_not_overwritten_by_new_episode():
    """Награда терминала принадлежит закончившемуся эпизоду, а не новому."""
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1}),
        ScriptedStep(
            decisions={20: 0.5},
            decision_rewards={20: 0.0},
            terminals={10: (0.9, 5.0, False)},
        ),
    ])
    vec.reset()
    result = vec.step(noop_actions(1))

    assert result.reward[0] == pytest.approx(5.0)


def test_actions_are_routed_to_the_right_agents():
    """Строки действий раскладываются по порядку агентов Unity, а не по слотам.

    Слот 0 — агент 10, слот 1 — агент 11. На третьем шаге Unity отдаёт решения
    в порядке (11, 20): действие агента 11 обязано прийти из строки его слота.
    """
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1, 11: 0.2}),
        ScriptedStep(decisions={11: 0.3}, terminals={10: (0.9, 1.0, False)}),
        ScriptedStep(decisions={11: 0.4, 20: 0.7}),
    ])
    vec.reset()
    vec.step(np.array([[1], [2]], dtype=np.int32))   # оба слота активны
    vec.step(np.array([[7], [3]], dtype=np.int32))   # активен только слот 1 (агент 11)

    fake = vec.handle.env
    assert fake.actions_seen[0].reshape(-1).tolist() == [1, 2]     # агенты 10, 11
    assert fake.actions_seen[1].reshape(-1).tolist() == [3]        # только агент 11


def test_step_rejects_action_array_of_wrong_length():
    vec = make_vec([ScriptedStep(decisions={10: 0.1, 11: 0.2})])
    vec.reset()
    with pytest.raises(ValueError, match="строку на каждый слот"):
        vec.step(noop_actions(1))


def test_reset_without_agents_is_an_error():
    vec = make_vec([ScriptedStep()])
    with pytest.raises(RuntimeError, match="ни одного агента"):
        vec.reset()


def test_more_agents_than_slots_is_an_error():
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1}),
        ScriptedStep(decisions={10: 0.2, 11: 0.3}),
    ])
    vec.reset()
    with pytest.raises(RuntimeError, match="свободных слотов нет"):
        vec.step(noop_actions(1))


def test_reset_is_skipped_when_nothing_was_stepped():
    """Повторный env.reset() в ML-Agents стоит шага с нулевым действием.

    Измерено на реальном билде: после второго подряд `reset()` агент
    GridWorld оказывается не в стартовой клетке (0,0), а в (0,1) —
    ровно один шаг действием 0. Поэтому обёртка не сбрасывает уже
    сброшенную среду (T-7 в docs/07_TROUBLESHOOTING.md).
    """
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1}),
        ScriptedStep(decisions={10: 0.2}),
    ])
    vec.handle.stepped_since_reset = False

    vec.reset()
    assert vec.handle.env.resets == 0, "сброс уже сброшенной среды выполнять нельзя"

    vec.reset()
    assert vec.handle.env.resets == 0


def test_reset_is_performed_after_a_step():
    vec = make_vec([
        ScriptedStep(decisions={10: 0.1}),
        ScriptedStep(decisions={10: 0.2}),
    ])
    vec.reset()
    vec.step(noop_actions(1))
    assert vec.handle.stepped_since_reset

    vec.reset()
    assert vec.handle.env.resets == 1
    assert not vec.handle.stepped_since_reset
