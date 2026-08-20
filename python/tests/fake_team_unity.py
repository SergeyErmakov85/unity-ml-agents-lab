"""Поддельная среда двух команд для тестов `labrl.envs.team_env`.

Поднять настоящий футбольный билд в тестах нельзя, а проверять нужно не
Unity, а **нашу** логику: группировку агентов по ``group_id``, сложение
командной награды из ``group_reward`` и личных, различение ``terminated``
и ``truncated`` и раскладку действий обратно по агентам.

Сценарий задаётся по шагам и по командам, как в :mod:`tests.fake_unity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from mlagents_envs.base_env import (
    ActionSpec,
    BehaviorSpec,
    DecisionSteps,
    DimensionProperty,
    ObservationSpec,
    ObservationType,
    TerminalSteps,
)


@dataclass
class AgentRecord:
    """Один агент на одном шаге."""

    agent_id: int
    group_id: int
    obs_value: float
    reward: float = 0.0
    group_reward: float = 0.0
    #: Только для терминальных записей.
    interrupted: bool = False


@dataclass
class TeamScriptedStep:
    """Шаг сценария: что отдаёт каждая команда."""

    #: ``team_id -> список агентов, ждущих действия``.
    decisions: dict[int, list[AgentRecord]] = field(default_factory=dict)
    #: ``team_id -> список агентов, завершивших эпизод``.
    terminals: dict[int, list[AgentRecord]] = field(default_factory=dict)


class FakeTeamUnityEnv:
    """Минимальная замена ``UnityEnvironment`` с двумя поведениями."""

    def __init__(
        self,
        env_id: str,
        obs_dim: int,
        spec: BehaviorSpec,
        script: Sequence[TeamScriptedStep],
        team_ids: tuple[int, ...] = (0, 1),
    ) -> None:
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.team_ids = team_ids
        self.behavior_specs = {f"{env_id}?team={t}": spec for t in team_ids}
        self.script = list(script)
        self.cursor = 0
        self.actions_seen: list[tuple[str, np.ndarray]] = []
        self.resets = 0
        self.closed = False

    # --- API, которое использует TeamUnityEnv ---------------------------

    def reset(self) -> None:
        self.resets += 1
        self.cursor = 0

    def step(self) -> None:
        self.cursor = min(self.cursor + 1, len(self.script) - 1)

    def set_actions(self, behavior_name: str, action_tuple) -> None:  # noqa: ANN001
        discrete = getattr(action_tuple, "discrete", None)
        self.actions_seen.append(
            (behavior_name, np.array(discrete) if discrete is not None else np.zeros((0, 0)))
        )

    def get_steps(self, behavior_name: str) -> tuple[DecisionSteps, TerminalSteps]:
        team = int(behavior_name.split("?team=")[1])
        step = self.script[self.cursor]
        return (
            self._make_decision(step.decisions.get(team, [])),
            self._make_terminal(step.terminals.get(team, [])),
        )

    def close(self) -> None:
        self.closed = True

    # --- сборка структур ------------------------------------------------

    def _obs(self, records: list[AgentRecord]) -> np.ndarray:
        if not records:
            return np.zeros((0, self.obs_dim), dtype=np.float32)
        return np.array(
            [[r.obs_value] * self.obs_dim for r in records], dtype=np.float32
        ).reshape(len(records), self.obs_dim)

    def _make_decision(self, records: list[AgentRecord]) -> DecisionSteps:
        return DecisionSteps(
            obs=[self._obs(records)],
            reward=np.array([r.reward for r in records], dtype=np.float32),
            agent_id=np.array([r.agent_id for r in records], dtype=np.int32),
            action_mask=None,
            group_id=np.array([r.group_id for r in records], dtype=np.int32),
            group_reward=np.array([r.group_reward for r in records], dtype=np.float32),
        )

    def _make_terminal(self, records: list[AgentRecord]) -> TerminalSteps:
        return TerminalSteps(
            obs=[self._obs(records)],
            reward=np.array([r.reward for r in records], dtype=np.float32),
            interrupted=np.array([r.interrupted for r in records], dtype=bool),
            agent_id=np.array([r.agent_id for r in records], dtype=np.int32),
            group_id=np.array([r.group_id for r in records], dtype=np.int32),
            group_reward=np.array([r.group_reward for r in records], dtype=np.float32),
        )


def make_team_spec(obs_dim: int, branches: tuple[int, ...] = (3, 3, 3)) -> BehaviorSpec:
    return BehaviorSpec(
        observation_specs=[
            ObservationSpec(
                shape=(obs_dim,),
                dimension_property=(DimensionProperty.NONE,),
                observation_type=ObservationType.DEFAULT,
                name="VectorSensor",
            )
        ],
        action_spec=ActionSpec(continuous_size=0, discrete_branches=branches),
    )


class FakeHandle:
    """Заглушка :class:`labrl.envs.unity_env.UnityEnvHandle`."""

    def __init__(self, env: FakeTeamUnityEnv) -> None:
        self.env = env
        self.stepped_since_reset = False

    def close(self) -> None:
        self.env.close()
