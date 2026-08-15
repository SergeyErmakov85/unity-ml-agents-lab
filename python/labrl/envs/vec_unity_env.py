"""Векторизация: K арен одного билда как K параллельных сред (требование 8.2).

Здесь живёт самое коварное место связки Python ↔ Unity, поэтому объясняю
модель подробно.

Как Unity отдаёт шаги
---------------------
``env.get_steps(behavior)`` возвращает **две** непересекающиеся выборки:

``DecisionSteps``
    агенты, которым нужно действие **сейчас**: их наблюдение — состояние,
    из которого предстоит действовать;

``TerminalSteps``
    агенты, чей эпизод **только что** закончился: их наблюдение — последнее
    состояние эпизода, а поле ``interrupted`` различает две причины конца.

Агент никогда не попадает в обе выборки одного шага. Более того,
**идентификатор агента меняется при каждом новом эпизоде** — он равен номеру
эпизода, а не номеру арены. Поэтому «слот» (индекс параллельной среды)
нельзя брать равным ``agent_id``: нужен собственный учёт, который здесь и
реализован через список свободных слотов.

terminated или truncated — почему это не придирка
-------------------------------------------------
``interrupted=True`` означает обрыв по ``MaxStep``: эпизод оборвали снаружи,
а не потому, что среда пришла в терминальное состояние. Ценность будущего
в этот момент **не равна нулю**, поэтому бутстрэппинг обязан выполняться::

    y = r + γ · V(s')      если truncated (обрыв по времени)
    y = r                  если terminated (истинное завершение)

Ошибка здесь не роняет обучение — она его тихо портит: агент учится тому,
что «время вышло» равносильно «всё потеряно», и начинает избегать длинных
траекторий. Инструкция (8.3) требует покрыть это тестом; тест —
``python/tests/test_vec_env_termination.py``.

Асинхронность и маска ``active``
--------------------------------
На шаге, когда слот получил терминал, новый эпизод в нём ещё не начат:
агент появится в ``DecisionSteps`` только следующим шагом. Такой слот
помечается ``active=False`` — из него **нельзя** брать наблюдение для политики,
и действие для него на следующем шаге не задаётся. Скрывать это от кода
обучения нельзя: любая попытка «дорисовать» отсутствующее наблюдение —
это выдумывание данных.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps

from labrl.envs.unity_env import UnityEnvHandle


@dataclass
class StepResult:
    """Результат одного шага по всем слотам.

    Два набора наблюдений — не избыточность, а разные состояния. ``obs`` —
    состояние, **из которого предстоит действовать**; ``final_obs`` — последнее
    состояние закончившегося эпизода, нужное для расчёта цели обновления.
    Если арена успела начать новый эпизод в том же шаге, ``obs`` уже содержит
    начальное наблюдение нового эпизода, а ``final_obs`` — терминальное старого.

    Attributes:
        obs: список массивов по сенсорам, каждый формы ``(N, *shape)``.
            Валидно там, где ``active``.
        final_obs: то же по формам; валидно там, где ``terminated | truncated``.
        reward: ``(N,)`` — награда за прошедший шаг.
        terminated: ``(N,)`` — эпизод завершился **истинно** (цель, провал).
        truncated: ``(N,)`` — эпизод оборван по ``MaxStep``.
        active: ``(N,)`` — слот ждёт действия; наблюдение в ``obs`` валидно.
        info: диагностика, включая метрики из Unity (``env_stats``).
    """

    obs: list[np.ndarray]
    final_obs: list[np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    active: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> np.ndarray:
        """Эпизод кончился по любой причине. Для бутстрэппинга использовать нельзя."""
        return self.terminated | self.truncated


class VecUnityEnv:
    """K одинаковых арен внутри одного билда как K параллельных сред.

    Args:
        handle: открытая среда (:func:`labrl.envs.unity_env.open_unity_env`).
        num_envs: ожидаемое число слотов. ``None`` — определить по числу
            агентов при первом сбросе.

    Гарантии API (требование 8.3):

    * ``reset() -> obs``;
    * ``step(actions) -> StepResult`` с раздельными ``terminated`` и ``truncated``.
    """

    def __init__(self, handle: UnityEnvHandle, num_envs: int | None = None) -> None:
        self.handle = handle
        self.spec = handle.spec
        self.behavior_name = handle.behavior_name

        self._num_envs = num_envs
        self._slot_of_agent: dict[int, int] = {}
        self._free_slots: list[int] = []
        self._obs: list[np.ndarray] = []
        self._pending: dict[int, int] = {}  # agent_id -> slot, ждут действия
        self._pending_order: list[int] = []  # порядок агентов, в каком их отдал Unity

    # --- свойства -------------------------------------------------------

    @property
    def num_envs(self) -> int:
        if self._num_envs is None:
            raise RuntimeError("число сред неизвестно до первого reset()")
        return self._num_envs

    @property
    def obs_shapes(self) -> list[tuple[int, ...]]:
        return self.handle.obs_shapes

    @property
    def action_spec(self):  # noqa: ANN201 — тип из mlagents_envs
        return self.spec.action_spec

    @property
    def single_obs_dim(self) -> int:
        """Суммарная размерность векторных наблюдений — вход политики.

        Определён только для одномерных сенсоров: для визуальных наблюдений
        политика принимает список тензоров, а не один вектор.
        """
        if any(len(shape) != 1 for shape in self.obs_shapes):
            raise ValueError(
                f"single_obs_dim определён только для векторных сенсоров, "
                f"формы: {self.obs_shapes}"
            )
        return sum(shape[0] for shape in self.obs_shapes)

    # --- жизненный цикл -------------------------------------------------

    def reset(self) -> list[np.ndarray]:
        """Сбрасывает среду и раздаёт слоты.

        Слоты назначаются по возрастанию ``agent_id``, чтобы при одинаковом
        сиде номер слота указывал на ту же арену от запуска к запуску.
        """
        self.handle.env.reset()
        decision, _ = self.handle.env.get_steps(self.behavior_name)

        if self._num_envs is None:
            self._num_envs = len(decision)
        elif len(decision) != self._num_envs:
            raise RuntimeError(
                f"после reset() среда вернула {len(decision)} агентов, "
                f"ожидалось {self._num_envs}"
            )
        if self._num_envs == 0:
            raise RuntimeError(
                f"среда не вернула ни одного агента поведения {self.behavior_name!r}: "
                "проверьте, что сцена содержит агентов и Behavior Type не Heuristic Only"
            )

        self._slot_of_agent = {int(a): i for i, a in enumerate(sorted(decision.agent_id))}
        self._free_slots = []
        self._obs = [np.zeros((self._num_envs, *shape), dtype=np.float32) for shape in self.obs_shapes]

        self._absorb_decision(decision)
        return [o.copy() for o in self._obs]

    def step(self, actions: np.ndarray) -> StepResult:
        """Применяет действия к активным слотам и продвигает симуляцию на шаг.

        Args:
            actions: ``(N, num_branches)`` целых — для дискретных действий,
                ``(N, continuous_size)`` вещественных — для непрерывных.
                Строки неактивных слотов игнорируются, но массив обязан быть
                полной длины ``N``: так индекс строки всегда равен номеру слота.

        Returns:
            :class:`StepResult`.
        """
        self._set_actions(actions)
        self.handle.env.step()
        decision, terminal = self.handle.env.get_steps(self.behavior_name)
        return self._collect(decision, terminal)

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "VecUnityEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- внутреннее -----------------------------------------------------

    def _set_actions(self, actions: np.ndarray) -> None:
        """Раскладывает действия из слотов обратно в порядок агентов Unity.

        Порядок строк в ``ActionTuple`` обязан совпадать с порядком
        ``decision_steps.agent_id``, а не с порядком слотов, — иначе арены
        получат чужие действия, и обучение будет выглядеть «почти работающим».
        """
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"actions должен содержать строку на каждый слот: ожидалось {self.num_envs}, "
                f"получено {actions.shape[0]}"
            )
        if not self._pending:
            return

        agent_ids = self._pending_agent_ids
        rows = np.array([self._pending[a] for a in agent_ids], dtype=np.int64)
        selected = actions[rows]

        action_tuple = ActionTuple()
        if self.action_spec.continuous_size > 0:
            action_tuple.add_continuous(selected.astype(np.float32).reshape(len(rows), -1))
        if self.action_spec.discrete_size > 0:
            action_tuple.add_discrete(selected.astype(np.int32).reshape(len(rows), -1))

        self.handle.env.set_actions(self.behavior_name, action_tuple)

    @property
    def _pending_agent_ids(self) -> list[int]:
        """Идентификаторы ждущих агентов в том порядке, в каком их отдал Unity."""
        return list(self._pending_order)

    def _absorb_decision(self, decision: DecisionSteps) -> None:
        """Записывает наблюдения агентов, ждущих действия, и раздаёт новые слоты."""
        self._pending = {}
        self._pending_order = []

        for index, agent_id in enumerate(decision.agent_id):
            agent_id = int(agent_id)
            slot = self._slot_of_agent.get(agent_id)
            if slot is None:
                if not self._free_slots:
                    raise RuntimeError(
                        f"агент {agent_id} появился, но свободных слотов нет: "
                        f"число агентов превысило {self.num_envs}"
                    )
                slot = self._free_slots.pop(0)
                self._slot_of_agent[agent_id] = slot

            for sensor, batch in enumerate(decision.obs):
                self._obs[sensor][slot] = batch[index]

            self._pending[agent_id] = slot
            self._pending_order.append(agent_id)

    def _collect(self, decision: DecisionSteps, terminal: TerminalSteps) -> StepResult:
        """Собирает результат шага: сперва терминалы, затем новые решения."""
        n = self.num_envs
        reward = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        final_obs = [np.zeros((n, *shape), dtype=np.float32) for shape in self.obs_shapes]

        # 1. Терминалы. Наблюдение эпизода уходит в final_obs — в self._obs его
        #    писать нельзя: если арена начнёт новый эпизод в этом же шаге,
        #    начальное наблюдение затрёт терминальное, и цель обновления
        #    посчитается по чужому состоянию.
        for index, agent_id in enumerate(terminal.agent_id):
            agent_id = int(agent_id)
            slot = self._slot_of_agent.pop(agent_id, None)
            if slot is None:
                # Агент завершил эпизод, ни разу не запросив решения. В корректно
                # собранной сцене такого не бывает; молча пропускать нельзя.
                raise RuntimeError(
                    f"терминал агента {agent_id}, которому не назначен слот; "
                    "вероятно, сцена содержит агентов, не участвовавших в reset()"
                )

            for sensor, batch in enumerate(terminal.obs):
                final_obs[sensor][slot] = batch[index]

            reward[slot] = terminal.reward[index]
            # interrupted == обрыв по MaxStep -> truncated (бутстрэппинг нужен);
            # иначе истинное терминальное состояние -> terminated.
            if bool(terminal.interrupted[index]):
                truncated[slot] = True
            else:
                terminated[slot] = True

            self._free_slots.append(slot)

        # 2. Решения. Награда берётся только у агентов, которые уже занимали слот:
        #    у агента нового эпизода награда относится к чужой траектории.
        for index, agent_id in enumerate(decision.agent_id):
            slot = self._slot_of_agent.get(int(agent_id))
            if slot is not None:
                reward[slot] = decision.reward[index]

        self._absorb_decision(decision)

        active = np.zeros(n, dtype=bool)
        if self._pending:
            active[list(self._pending.values())] = True

        info: dict[str, Any] = {
            "env_stats": self.handle.channels.drain_stats(),
            "num_decisions": len(decision),
            "num_terminals": len(terminal),
        }

        return StepResult(
            obs=[o.copy() for o in self._obs],
            final_obs=final_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            active=active,
            info=info,
        )
