"""Поддельная среда Unity для тестов векторизации.

Настоящую среду в тестах поднять нельзя: нужен билд и запущенный Unity.
Но проверять нужно не Unity, а **нашу** логику разбора шагов — раскладку
агентов по слотам и различение ``terminated`` / ``truncated``. Для этого
достаточно объекта, отдающего те же ``DecisionSteps`` / ``TerminalSteps``,
что и настоящая среда.

Поддельная среда работает по сценарию: список шагов, где каждый шаг говорит,
какие агенты получили решение, а какие завершили эпизод и по какой причине.
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
class ScriptedStep:
    """Один шаг сценария.

    Attributes:
        decisions: ``agent_id -> значение наблюдения`` для агентов, ждущих действия.
        terminals: ``agent_id -> (значение наблюдения, награда, interrupted)``.
        decision_rewards: награды агентов из ``decisions``; по умолчанию 0.
    """

    decisions: dict[int, float] = field(default_factory=dict)
    terminals: dict[int, tuple[float, float, bool]] = field(default_factory=dict)
    decision_rewards: dict[int, float] = field(default_factory=dict)


class FakeUnityEnv:
    """Минимальная замена ``UnityEnvironment`` для тестов."""

    def __init__(self, behavior_name: str, obs_dim: int, spec: BehaviorSpec, script: Sequence[ScriptedStep]) -> None:
        self.behavior_name = behavior_name
        self.obs_dim = obs_dim
        self.behavior_specs = {behavior_name: spec}
        self.script = list(script)
        self.cursor = -1  # -1 = состояние после reset()
        self.actions_seen: list[np.ndarray] = []
        self.closed = False

    # --- API, которое использует VecUnityEnv ---------------------------

    def reset(self) -> None:
        self.cursor = 0

    def step(self) -> None:
        self.cursor += 1

    def set_actions(self, behavior_name: str, action_tuple) -> None:  # noqa: ANN001
        assert behavior_name == self.behavior_name
        discrete = getattr(action_tuple, "discrete", None)
        self.actions_seen.append(np.array(discrete) if discrete is not None else np.array([]))

    def get_steps(self, behavior_name: str) -> tuple[DecisionSteps, TerminalSteps]:
        assert behavior_name == self.behavior_name
        step = self.script[self.cursor]
        return self._make_decision(step), self._make_terminal(step)

    def close(self) -> None:
        self.closed = True

    # --- сборка структур ------------------------------------------------

    def _make_decision(self, step: ScriptedStep) -> DecisionSteps:
        ids = sorted(step.decisions)
        obs = np.array([[step.decisions[i]] * self.obs_dim for i in ids], dtype=np.float32).reshape(len(ids), self.obs_dim)
        reward = np.array([step.decision_rewards.get(i, 0.0) for i in ids], dtype=np.float32)
        return DecisionSteps(
            obs=[obs],
            reward=reward,
            agent_id=np.array(ids, dtype=np.int32),
            action_mask=None,
            group_id=np.zeros(len(ids), dtype=np.int32),
            group_reward=np.zeros(len(ids), dtype=np.float32),
        )

    def _make_terminal(self, step: ScriptedStep) -> TerminalSteps:
        ids = sorted(step.terminals)
        obs = np.array([[step.terminals[i][0]] * self.obs_dim for i in ids], dtype=np.float32).reshape(len(ids), self.obs_dim)
        reward = np.array([step.terminals[i][1] for i in ids], dtype=np.float32)
        interrupted = np.array([step.terminals[i][2] for i in ids], dtype=bool)
        return TerminalSteps(
            obs=[obs],
            reward=reward,
            interrupted=interrupted,
            agent_id=np.array(ids, dtype=np.int32),
            group_id=np.zeros(len(ids), dtype=np.int32),
            group_reward=np.zeros(len(ids), dtype=np.float32),
        )


def make_spec(obs_dim: int, branches: tuple[int, ...] = (4,)) -> BehaviorSpec:
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


class FakeChannels:
    """Заглушка боковых каналов: метрик из среды в тестах нет."""

    def drain_stats(self) -> dict[str, float]:
        return {}
