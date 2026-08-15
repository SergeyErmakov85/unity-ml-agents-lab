"""Подключение к среде Unity в двух режимах (требование 8.1).

**Editor-режим** (``file_name=None``) — отладка. Python ждёт, пока пользователь
нажмёт Play в редакторе. Видно, что делает агент; скорость обычная.

**Build-режим** (``file_name=builds/...``) — обучение. Билд запускается
с ``no_graphics=True`` и ``time_scale=20``, что даёт на порядок больше шагов
в секунду.

Оба режима дают одинаковый объект :class:`UnityEnvHandle`, поэтому код обучения
их не различает: меняется одна строка конфига.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.environment import UnityEnvironment

from labrl.envs.side_channels import LabSideChannels
from labrl.utils.config import resolve_path

#: Сколько секунд ждать подключения в Editor-режиме. Значение по умолчанию
#: mlagents-envs (60 с) слишком мало: пользователь должен успеть переключиться
#: в редактор и нажать Play.
EDITOR_TIMEOUT_SECONDS = 300


@dataclass
class UnityEnvHandle:
    """Открытая среда Unity вместе с её боковыми каналами и спецификацией.

    Не имеет методов шага: их предоставляет :class:`labrl.envs.vec_unity_env.VecUnityEnv`.
    Разделение сознательное — подключение и векторизация решают разные задачи,
    и подключение полезно само по себе (ноутбук ``00_setup_check``).
    """

    env: UnityEnvironment
    channels: LabSideChannels
    behavior_name: str
    spec: BehaviorSpec

    @property
    def obs_shapes(self) -> list[tuple[int, ...]]:
        """Формы наблюдений по сенсорам, без батча, в порядке регистрации в Unity.

        Тот же порядок обязан соблюдаться во входах ``obs_0``, ``obs_1``, …
        экспортированной ONNX-модели (контракт §1).
        """
        return [tuple(o.shape) for o in self.spec.observation_specs]

    @property
    def obs_names(self) -> list[str]:
        """Имена сенсоров из Unity — помогают понять, что за наблюдение под каким индексом."""
        return [o.name for o in self.spec.observation_specs]

    def describe(self) -> str:
        """Человекочитаемое описание пространств — для ячейки 4 ноутбука."""
        action = self.spec.action_spec
        lines = [
            f"behavior: {self.behavior_name}",
            f"наблюдения ({len(self.obs_shapes)} сенсоров):",
        ]
        for i, (shape, name) in enumerate(zip(self.obs_shapes, self.obs_names)):
            lines.append(f"  obs_{i}: shape={shape}  имя сенсора: {name!r}")
        lines.append(f"действия: непрерывных {action.continuous_size}, "
                     f"дискретные ветки {tuple(action.discrete_branches)}")
        return "\n".join(lines)

    def env_info(self) -> dict[str, Any]:
        """Словарь для ``env_info.json`` каталога прогона (требование 11.1)."""
        return {
            "behavior_name": self.behavior_name,
            "obs_shapes": [list(s) for s in self.obs_shapes],
            "obs_names": self.obs_names,
            "continuous_size": self.spec.action_spec.continuous_size,
            "discrete_branches": list(self.spec.action_spec.discrete_branches),
        }

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "UnityEnvHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_unity_env(
    env_id: str,
    build_path: str | Path | None = None,
    seed: int = 0,
    time_scale: float = 20.0,
    no_graphics: bool = True,
    worker_id: int = 0,
    env_parameters: Mapping[str, float] | None = None,
    num_areas: int = 1,
) -> UnityEnvHandle:
    """Открывает среду Unity и проверяет, что она та, которую ждут.

    Args:
        env_id: идентификатор среды ``E##_<Name>``. Он же Behavior Name (правило 5.3).
        build_path: путь к ``.exe`` билда. ``None`` — Editor-режим: Python ждёт
            нажатия Play в редакторе.
        seed: сид. Передаётся и в ``UnityEnvironment`` (влияет на генератор Unity),
            и через ``EnvironmentParametersChannel`` (его читают арены, требование 7.3).
        time_scale: множитель скорости симуляции. В Editor-режиме игнорируется
            и заменяется на 1.0: разогнанный редактор бесполезен для отладки.
        no_graphics: рендерить ли билд. В Editor-режиме не применяется.
        worker_id: смещение порта. Разные значения позволяют запустить
            несколько сред параллельно; для Editor-режима обязан быть 0.
        env_parameters: дополнительные параметры среды из конфига.
        num_areas: число тренировочных арен, запрашиваемое у билда.

    Returns:
        :class:`UnityEnvHandle`.

    Raises:
        FileNotFoundError: билд не найден по указанному пути.
        KeyError: в среде нет поведения с именем ``env_id`` — почти всегда это
            означает расхождение Behavior Name в Unity и идентификатора среды.
    """
    editor_mode = build_path is None

    file_name: str | None = None
    if not editor_mode:
        exe = resolve_path(build_path)
        if not exe.is_file():
            raise FileNotFoundError(
                f"билд не найден: {exe}\n"
                f"Соберите среду: python scripts/build_env.py {env_id}"
            )
        file_name = str(exe)

    channels = LabSideChannels()

    if editor_mode:
        print("Editor-режим: нажмите Play в Unity — ожидание подключения…", flush=True)

    env = UnityEnvironment(
        file_name=file_name,
        worker_id=worker_id,
        seed=seed,
        no_graphics=no_graphics if not editor_mode else False,
        timeout_wait=EDITOR_TIMEOUT_SECONDS if editor_mode else 60,
        side_channels=channels.as_list(),
        num_areas=num_areas,
    )

    # Каналы работают только после установления связи, поэтому настройка идёт
    # после конструктора, а reset() ниже доносит параметры до среды.
    channels.configure_engine(time_scale=1.0 if editor_mode else time_scale)
    channels.set_seed(seed)
    if env_parameters:
        channels.set_parameters(env_parameters)

    env.reset()

    try:
        behavior_key = resolve_behavior_name(env_id, list(env.behavior_specs))
    except KeyError:
        env.close()
        raise

    return UnityEnvHandle(
        env=env,
        channels=channels,
        behavior_name=behavior_key,
        spec=env.behavior_specs[behavior_key],
    )


def resolve_behavior_name(env_id: str, available: list[str]) -> str:
    """Находит полное имя поведения, соответствующее идентификатору среды.

    Unity отдаёт **полное** имя поведения — `BehaviorName?team=<TeamId>`
    (`BehaviorParameters.FullyQualifiedBehaviorName`), поэтому среда с
    Behavior Name `E01_GridWorld` и TeamId 0 видна из Python как
    ``E01_GridWorld?team=0``. Идентификатор среды при этом остаётся без
    суффикса: команда — свойство сцены, а не примера.

    Args:
        env_id: идентификатор среды `E##_<Name>`.
        available: ключи `env.behavior_specs`.

    Returns:
        Ключ, под которым поведение зарегистрировано в среде.

    Raises:
        KeyError: подходящего поведения нет, либо их несколько (несколько
            команд одного поведения — случай self-play, который обязан
            обрабатываться явно, а не выбором «первого попавшегося»).
    """
    matches = [name for name in available if name == env_id or name.startswith(env_id + "?team=")]

    if not matches:
        raise KeyError(
            f"в среде нет поведения {env_id!r}; доступны: {available}.\n"
            "Behavior Name в Unity обязан совпадать с идентификатором среды (правило 5.3): "
            "проверьте BehaviorParameters на агенте или пересоберите сцену Setup-скриптом."
        )
    if len(matches) > 1:
        raise KeyError(
            f"идентификатору {env_id!r} соответствует несколько поведений: {matches}. "
            "Несколько команд одного поведения (self-play) требуют явного выбора — "
            "передайте полное имя вида 'E##_Name?team=N'."
        )
    return matches[0]
