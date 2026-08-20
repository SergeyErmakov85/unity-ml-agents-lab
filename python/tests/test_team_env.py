"""Обёртка двух команд: группировка, командная награда, terminated/truncated.

Проверяется наша логика разбора шагов, а не Unity. Три класса ошибок,
каждый из которых обучение не роняет, а тихо портит:

1. **Командная награда сложена неверно.** ``group_reward`` одинаков у всех
   членов группы; сложить его n раз значит удвоить (при n = 2) цену гола
   относительно платы за время.
2. **Агенты перепутаны между группами.** Наблюдение игрока одной арены
   попало бы в шаг команды другой — критик учился бы на несуществующих
   состояниях.
3. **Обрыв по времени принят за истинное завершение** (требование 8.3).
"""

from __future__ import annotations

import numpy as np
import pytest

from labrl.envs.team_env import TeamUnityEnv, team_behavior_name
from tests.fake_team_unity import (
    AgentRecord,
    FakeHandle,
    FakeTeamUnityEnv,
    TeamScriptedStep,
    make_team_spec,
)

ENV_ID = "E08_SoccerArena"
OBS_DIM = 4


def build(script: list[TeamScriptedStep], team_size: int = 2) -> TeamUnityEnv:
    fake = FakeTeamUnityEnv(ENV_ID, OBS_DIM, make_team_spec(OBS_DIM), script)
    return TeamUnityEnv(FakeHandle(fake), ENV_ID, team_size=team_size)


def both_teams(records_west: list[AgentRecord], records_east: list[AgentRecord]) -> TeamScriptedStep:
    return TeamScriptedStep(decisions={0: records_west, 1: records_east})


# --- имена поведений ----------------------------------------------------


def test_behavior_name_format():
    """Формат задан Unity, а не нами: BehaviorName + '?team=' + TeamId."""
    assert team_behavior_name("E08_SoccerArena", 1) == "E08_SoccerArena?team=1"


def test_missing_team_raises():
    """Отсутствие второй команды — ошибка сцены, а не повод молча продолжить."""
    fake = FakeTeamUnityEnv(ENV_ID, OBS_DIM, make_team_spec(OBS_DIM), [TeamScriptedStep()])
    del fake.behavior_specs[team_behavior_name(ENV_ID, 1)]
    with pytest.raises(KeyError, match="TeamId"):
        TeamUnityEnv(FakeHandle(fake), ENV_ID, team_size=2)


# --- группировка --------------------------------------------------------


def test_groups_are_separated_by_group_id():
    """Две арены = две группы; наблюдения не перемешиваются."""
    step = both_teams(
        [
            AgentRecord(agent_id=10, group_id=100, obs_value=1.0),
            AgentRecord(agent_id=11, group_id=100, obs_value=2.0),
            AgentRecord(agent_id=12, group_id=200, obs_value=3.0),
            AgentRecord(agent_id=13, group_id=200, obs_value=4.0),
        ],
        [
            AgentRecord(agent_id=20, group_id=101, obs_value=-1.0),
            AgentRecord(agent_id=21, group_id=101, obs_value=-2.0),
            AgentRecord(agent_id=22, group_id=201, obs_value=-3.0),
            AgentRecord(agent_id=23, group_id=201, obs_value=-4.0),
        ],
    )
    env = build([step])
    out = env.reset()

    assert env.num_groups == 2
    west = out[0]
    assert west.obs.shape == (2, 2, OBS_DIM)
    # Порядок внутри группы — по agent_id.
    assert west.obs[0, 0, 0] == pytest.approx(1.0)
    assert west.obs[0, 1, 0] == pytest.approx(2.0)
    assert west.obs[1, 0, 0] == pytest.approx(3.0)
    assert west.active.all()
    assert west.awaiting.all()

    east = out[1]
    assert east.obs[0, 0, 0] == pytest.approx(-1.0)
    assert east.obs[1, 1, 0] == pytest.approx(-4.0)


# --- командная награда --------------------------------------------------


def test_team_reward_counts_group_bonus_once():
    """r_команды = group_reward (один раз) + Σ личных наград."""
    step = both_teams(
        [
            AgentRecord(agent_id=10, group_id=100, obs_value=0.0, reward=-0.01, group_reward=1.0),
            AgentRecord(agent_id=11, group_id=100, obs_value=0.0, reward=-0.02, group_reward=1.0),
        ],
        [AgentRecord(agent_id=20, group_id=101, obs_value=0.0),
         AgentRecord(agent_id=21, group_id=101, obs_value=0.0)],
    )
    out = build([step]).reset()
    # 1.0 (командная, один раз) + (−0.01) + (−0.02)
    assert out[0].reward[0] == pytest.approx(0.97, abs=1e-6)


# --- завершение ---------------------------------------------------------


def test_terminated_and_truncated_are_distinguished():
    """Гол — истинное завершение; обрыв по времени — truncated (8.3)."""
    goal = TeamScriptedStep(
        terminals={
            0: [
                AgentRecord(agent_id=10, group_id=100, obs_value=9.0, group_reward=1.0, interrupted=False),
                AgentRecord(agent_id=11, group_id=100, obs_value=9.0, group_reward=1.0, interrupted=False),
            ],
            1: [
                AgentRecord(agent_id=20, group_id=101, obs_value=9.0, group_reward=-1.0, interrupted=False),
                AgentRecord(agent_id=21, group_id=101, obs_value=9.0, group_reward=-1.0, interrupted=False),
            ],
        }
    )
    timeout = TeamScriptedStep(
        terminals={
            0: [
                AgentRecord(agent_id=12, group_id=100, obs_value=7.0, interrupted=True),
                AgentRecord(agent_id=13, group_id=100, obs_value=7.0, interrupted=True),
            ],
            1: [
                AgentRecord(agent_id=22, group_id=101, obs_value=7.0, interrupted=True),
                AgentRecord(agent_id=23, group_id=101, obs_value=7.0, interrupted=True),
            ],
        }
    )
    env = build([goal, timeout])
    out = env.reset()

    assert out[0].terminated[0] and not out[0].truncated[0]
    assert out[0].final_active[0].all()
    assert out[0].final_obs[0, 0, 0] == pytest.approx(9.0)
    # Пропустившая команда получает свой минус ровно один раз.
    assert out[1].reward[0] == pytest.approx(-1.0, abs=1e-6)

    after = env.step({0: np.zeros((1, 2, 3), dtype=np.int64), 1: np.zeros((1, 2, 3), dtype=np.int64)})
    assert after[0].truncated[0] and not after[0].terminated[0]
    assert after[0].done[0]


def test_group_not_awaiting_when_only_terminals():
    """Группа, отдавшая только терминалы, действий не ждёт: писать в неё
    наблюдение из obs нельзя — оно нулевое."""
    step = TeamScriptedStep(
        terminals={
            0: [AgentRecord(agent_id=10, group_id=100, obs_value=1.0, interrupted=True),
                AgentRecord(agent_id=11, group_id=100, obs_value=1.0, interrupted=True)],
            1: [AgentRecord(agent_id=20, group_id=101, obs_value=1.0, interrupted=True),
                AgentRecord(agent_id=21, group_id=101, obs_value=1.0, interrupted=True)],
        }
    )
    out = build([step]).reset()
    assert not out[0].awaiting[0]
    assert not out[0].active[0].any()


# --- раскладка действий -------------------------------------------------


def test_no_actions_are_sent_when_nobody_awaits():
    """Команде без ждущих агентов действия не задаются вовсе.

    Пустой ``ActionTuple()`` здесь не годится: у него ``continuous is None``,
    и ``ActionSpec._validate_action`` падает на обращении к ``.shape``.
    Ошибка обнаружена на живом билде `E08_SoccerArena` в момент, когда
    один матч закончился, а второй ещё шёл.
    """
    step = TeamScriptedStep(
        decisions={0: [AgentRecord(agent_id=10, group_id=100, obs_value=1.0),
                       AgentRecord(agent_id=11, group_id=100, obs_value=1.0)]},
        terminals={1: [AgentRecord(agent_id=20, group_id=101, obs_value=1.0, interrupted=False),
                       AgentRecord(agent_id=21, group_id=101, obs_value=1.0, interrupted=False)]},
    )
    env = build([step, step])
    env.reset()
    env.step({0: np.zeros((1, 2, 3), dtype=np.int64), 1: np.zeros((1, 2, 3), dtype=np.int64)})

    sent = [name for name, _ in env.handle.env.actions_seen]
    assert team_behavior_name(ENV_ID, 0) in sent
    assert team_behavior_name(ENV_ID, 1) not in sent


def test_actions_are_routed_to_the_right_agents():
    """Действие группы 1 не должно достаться агентам группы 0."""
    step = both_teams(
        [
            AgentRecord(agent_id=10, group_id=100, obs_value=1.0),
            AgentRecord(agent_id=11, group_id=100, obs_value=1.0),
            AgentRecord(agent_id=12, group_id=200, obs_value=2.0),
            AgentRecord(agent_id=13, group_id=200, obs_value=2.0),
        ],
        [
            AgentRecord(agent_id=20, group_id=101, obs_value=0.0),
            AgentRecord(agent_id=21, group_id=101, obs_value=0.0),
            AgentRecord(agent_id=22, group_id=201, obs_value=0.0),
            AgentRecord(agent_id=23, group_id=201, obs_value=0.0),
        ],
    )
    env = build([step, step])
    env.reset()

    west = np.zeros((2, 2, 3), dtype=np.int64)
    west[0, 0] = [1, 0, 0]
    west[0, 1] = [0, 1, 0]
    west[1, 0] = [0, 0, 1]
    west[1, 1] = [2, 2, 2]
    env.step({0: west, 1: np.zeros((2, 2, 3), dtype=np.int64)})

    sent = dict(env.handle.env.actions_seen)[team_behavior_name(ENV_ID, 0)]
    # Порядок в Unity — по возрастанию agent_id: 10, 11, 12, 13.
    assert sent.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]]
