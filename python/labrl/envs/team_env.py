"""Мультиагентная среда: две команды в одном билде (`E08_SoccerArena`).

Чем это отличается от :mod:`labrl.envs.vec_unity_env`
-----------------------------------------------------
Векторизованная обёртка трактует каждого агента как отдельную параллельную
среду. Для футбола это неверно дважды:

1. **Четверо игроков арены — не четыре среды, а одна.** Они видят один мяч
   и влияют друг на друга; сложить их в независимые слоты значит выбросить
   ровно ту связь, ради которой среда и заведена.
2. **Команд две, и они видны Python как разные поведения.** Unity добавляет
   к имени поведения идентификатор команды
   (``BehaviorParameters.FullyQualifiedBehaviorName``), поэтому одна сцена
   отдаёт ``E08_SoccerArena?team=0`` и ``E08_SoccerArena?team=1``. Это и есть
   механизм self-play: две команды с одинаковым Behavior Name, но разными
   ``TeamId``, каждой можно назначить свою политику.

Единица шага здесь — **команда**: наблюдения её игроков, их действия и одна
общая награда.

Как считается награда команды
-----------------------------
Низкоуровневый API отдаёт два поля::

    reward        личная награда агента с прошлого решения
    group_reward  награда, начисленная группе через SimpleMultiAgentGroup

``group_reward`` одинаков у всех членов группы, поэтому берётся **один раз**,
а личные складываются::

    r_команды = group_reward + Σ_i reward_i

Группировка агентов
-------------------
По полю ``group_id``: его выдаёт `SimpleMultiAgentGroup`, и оно постоянно
на всё время жизни группы, тогда как ``agent_id`` меняется с каждым новым
эпизодом. Внутри группы игроки упорядочиваются по ``agent_id`` — порядок
не имеет значения (внимание в MA-POCA перестановочно инвариантно), но
обязан быть одинаковым для наблюдений и действий одного шага.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps

from labrl.envs.unity_env import UnityEnvHandle


def team_behavior_name(env_id: str, team_id: int) -> str:
    """Полное имя поведения команды: ``E08_SoccerArena?team=0``.

    Формат задан Unity (`BehaviorParameters.FullyQualifiedBehaviorName`,
    `com.unity.ml-agents@4.0.3`), а не выбран здесь.
    """
    return f"{env_id}?team={int(team_id)}"


@dataclass
class TeamStep:
    """Состояние одной команды на одном шаге.

    Attributes:
        obs: ``(G, n, obs_dim)`` — наблюдения игроков каждой группы.
        active: ``(G, n)`` — 1 там, где игрок присутствует и ждёт действия.
        reward: ``(G,)`` — командная награда за прошедший шаг.
        terminated: ``(G,)`` — матч завершился истинно (гол).
        truncated: ``(G,)`` — матч оборван по времени.
        final_obs: ``(G, n, obs_dim)`` — последние наблюдения завершившегося
            матча; валидно там, где ``terminated | truncated``.
        final_active: ``(G, n)`` — маска к ``final_obs``.
        awaiting: ``(G,)`` — группа ждёт действий; в ``obs`` валидные данные.
    """

    obs: np.ndarray
    active: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_obs: np.ndarray
    final_active: np.ndarray
    awaiting: np.ndarray

    @property
    def done(self) -> np.ndarray:
        """Матч кончился по любой причине. Для бутстрэппинга использовать нельзя."""
        return self.terminated | self.truncated


class TeamUnityEnv:
    """Две команды одного билда как два потока групповых шагов.

    Args:
        handle: открытая среда. ``handle.behavior_name`` может указывать
            на любую из команд — обёртка обращается к обеим по полным именам.
        env_id: идентификатор среды без суффикса команды, например
            ``E08_SoccerArena``.
        team_size: сколько игроков в команде.
        num_groups: сколько групп ожидается на команду (обычно = числу арен).
            ``None`` — определить по первому сбросу.

    Гарантии: ``reset()`` возвращает словарь ``{team_id: TeamStep}``,
    ``step(actions)`` принимает такой же словарь действий и возвращает
    такой же словарь результатов. Различие ``terminated`` / ``truncated``
    сохраняется (требование 8.3).
    """

    def __init__(
        self,
        handle: UnityEnvHandle,
        env_id: str,
        team_size: int = 2,
        num_groups: int | None = None,
        team_ids: tuple[int, ...] = (0, 1),
    ) -> None:
        self.handle = handle
        self.env = handle.env
        self.env_id = env_id
        self.team_size = int(team_size)
        self.team_ids = tuple(int(t) for t in team_ids)

        available = list(self.env.behavior_specs)
        self.behavior_of: dict[int, str] = {}
        for team in self.team_ids:
            name = team_behavior_name(env_id, team)
            if name not in available:
                raise KeyError(
                    f"в среде нет поведения {name!r}; доступны: {available}.\n"
                    "Проверьте TeamId в BehaviorParameters: команда 0 и команда 1 "
                    "обязаны иметь одинаковый Behavior Name и разные TeamId "
                    "(ENV_SPEC.md среды, §5)."
                )
            self.behavior_of[team] = name

        self.spec = self.env.behavior_specs[self.behavior_of[self.team_ids[0]]]
        self._num_groups = num_groups
        # group_id -> индекс группы. Заполняется при первом появлении группы.
        self._group_index: dict[int, dict[int, int]] = {t: {} for t in self.team_ids}
        # Порядок агентов внутри группы на текущем шаге: (team, group) -> [agent_id].
        self._order: dict[tuple[int, int], list[int]] = {}

    # --- свойства -------------------------------------------------------

    @property
    def num_groups(self) -> int:
        if self._num_groups is None:
            raise RuntimeError("число групп неизвестно до первого reset()")
        return self._num_groups

    @property
    def obs_dim(self) -> int:
        """Суммарная размерность векторных наблюдений — вход политики."""
        total = 0
        for spec in self.spec.observation_specs:
            if len(spec.shape) != 1:
                raise ValueError(
                    f"сенсор {spec.name!r} формы {spec.shape} не одномерный; "
                    "склейка наблюдений определена только для векторных сенсоров"
                )
            total += int(spec.shape[0])
        return total

    @property
    def action_spec(self):  # noqa: ANN201 — тип из mlagents_envs
        return self.spec.action_spec

    @property
    def discrete_branches(self) -> tuple[int, ...]:
        return tuple(int(b) for b in self.spec.action_spec.discrete_branches)

    # --- цикл -----------------------------------------------------------

    def reset(self) -> dict[int, TeamStep]:
        """Сбрасывает среду и возвращает начальные шаги обеих команд."""
        if self.handle.stepped_since_reset:
            self.env.reset()
            self.handle.stepped_since_reset = False
        return self._collect()

    def step(self, actions: dict[int, np.ndarray]) -> dict[int, TeamStep]:
        """Задаёт действия обеим командам и продвигает среду на шаг.

        Args:
            actions: ``{team_id: (G, n, num_branches)}`` — индексы дискретных
                веток. Значения для групп, у которых ``awaiting = False``,
                игнорируются: такой группе Unity действие не запрашивал.
        """
        for team in self.team_ids:
            self._set_actions(team, actions[team])

        self.env.step()
        self.handle.stepped_since_reset = True
        return self._collect()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "TeamUnityEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- внутреннее -----------------------------------------------------

    def _slot(self, team: int, group_id: int) -> int:
        """Индекс группы; новая группа получает следующий свободный индекс."""
        index = self._group_index[team]
        if group_id not in index:
            index[group_id] = len(index)
        return index[group_id]

    def _ensure_num_groups(self) -> int:
        if self._num_groups is None:
            sizes = {t: len(self._group_index[t]) for t in self.team_ids}
            if len(set(sizes.values())) != 1:
                raise RuntimeError(
                    f"у команд разное число групп: {sizes}. Сцена обязана содержать "
                    "по одной группе на арену у каждой команды."
                )
            self._num_groups = next(iter(sizes.values()))
        return self._num_groups

    def _collect(self) -> dict[int, TeamStep]:
        """Читает шаги обеих команд и раскладывает их по группам."""
        raw: dict[int, tuple[DecisionSteps, TerminalSteps]] = {}
        for team in self.team_ids:
            decision, terminal = self.env.get_steps(self.behavior_of[team])
            raw[team] = (decision, terminal)
            for group_id in np.unique(np.concatenate([decision.group_id, terminal.group_id])
                                      if len(decision) + len(terminal) else np.zeros(0, dtype=np.int32)):
                self._slot(team, int(group_id))

        groups = self._ensure_num_groups()
        return {team: self._build_step(team, *raw[team], groups) for team in self.team_ids}

    def _build_step(
        self, team: int, decision: DecisionSteps, terminal: TerminalSteps, groups: int
    ) -> TeamStep:
        n = self.team_size
        dim = self.obs_dim

        obs = np.zeros((groups, n, dim), dtype=np.float32)
        active = np.zeros((groups, n), dtype=bool)
        reward = np.zeros(groups, dtype=np.float32)
        terminated = np.zeros(groups, dtype=bool)
        truncated = np.zeros(groups, dtype=bool)
        final_obs = np.zeros((groups, n, dim), dtype=np.float32)
        final_active = np.zeros((groups, n), dtype=bool)
        awaiting = np.zeros(groups, dtype=bool)

        # Командная награда: group_reward берётся ОДИН раз на группу
        # (он одинаков у всех членов), личные складываются.
        group_bonus: dict[int, float] = {}

        def accumulate(steps: Any, index: int) -> int:
            slot = self._slot(team, int(steps.group_id[index]))
            reward[slot] += float(steps.reward[index])
            group_bonus[slot] = float(steps.group_reward[index])
            return slot

        # --- завершившиеся матчи ---
        terminal_order: dict[int, list[int]] = {}
        for i in range(len(terminal)):
            slot = accumulate(terminal, i)
            terminal_order.setdefault(slot, []).append(i)
            if terminal.interrupted[i]:
                truncated[slot] = True
            else:
                terminated[slot] = True

        for slot, indices in terminal_order.items():
            indices.sort(key=lambda i: int(terminal.agent_id[i]))
            for position, i in enumerate(indices[:n]):
                final_obs[slot, position] = self._flat_obs(terminal.obs, i)
                final_active[slot, position] = True

        # --- матчи, ждущие действия ---
        self._order = {k: v for k, v in self._order.items() if k[0] != team}
        decision_order: dict[int, list[int]] = {}
        for i in range(len(decision)):
            slot = accumulate(decision, i)
            decision_order.setdefault(slot, []).append(i)

        for slot, indices in decision_order.items():
            indices.sort(key=lambda i: int(decision.agent_id[i]))
            awaiting[slot] = True
            self._order[(team, slot)] = [int(decision.agent_id[i]) for i in indices[:n]]
            for position, i in enumerate(indices[:n]):
                obs[slot, position] = self._flat_obs(decision.obs, i)
                active[slot, position] = True

        for slot, bonus in group_bonus.items():
            reward[slot] += bonus

        return TeamStep(
            obs=obs,
            active=active,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            final_obs=final_obs,
            final_active=final_active,
            awaiting=awaiting,
        )

    @staticmethod
    def _flat_obs(obs_list: list[np.ndarray], index: int) -> np.ndarray:
        """Склейка наблюдений всех сенсоров одного агента в один вектор.

        Порядок сенсоров задаёт Unity, и он же обязан соблюдаться во входах
        ``obs_0``, ``obs_1``, … графа ONNX (контракт §1). Здесь склейка
        повторяет ``labrl.export.onnx_export._concat_obs`` — то же самое
        делает и обёртка экспорта, иначе Python и Unity видели бы разный вход.
        """
        return np.concatenate([np.asarray(o[index], dtype=np.float32).ravel() for o in obs_list])

    def _set_actions(self, team: int, actions: np.ndarray) -> None:
        """Раскладывает действия групп обратно по агентам Unity."""
        behavior = self.behavior_of[team]
        decision, _ = self.env.get_steps(behavior)
        if len(decision) == 0:
            # Ни один агент команды не ждёт решения — действий не задаём вовсе.
            # Пустой ActionTuple() здесь не подходит: у него continuous is None,
            # и ActionSpec._validate_action падает на обращении к .shape.
            # UnityEnvironment.step() сам подставит пустое действие поведению,
            # для которого действий не задано (mlagents_envs/environment.py).
            return

        actions = np.asarray(actions, dtype=np.int32)
        branches = len(self.discrete_branches)
        # Unity ждёт действия строго в порядке decision.agent_id — том же,
        # в каком он их отдал. Сопоставление идёт через таблицу порядка,
        # заполненную при чтении шага.
        by_agent: dict[int, np.ndarray] = {}
        for (t, slot), agent_ids in self._order.items():
            if t != team:
                continue
            for position, agent_id in enumerate(agent_ids):
                by_agent[agent_id] = actions[slot, position]

        out = np.zeros((len(decision), branches), dtype=np.int32)
        for i in range(len(decision)):
            agent_id = int(decision.agent_id[i])
            if agent_id in by_agent:
                out[i] = by_agent[agent_id]
        self.env.set_actions(behavior, ActionTuple(discrete=out))
