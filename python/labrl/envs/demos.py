"""Демонстрации эксперта: запись, хранение, чтение.

Что такое демонстрация
----------------------
Набор пар «наблюдение → действие», снятых с эксперта. Награда в них
**не нужна**: имитационное обучение по определению учится тому, *что* делает
эксперт, а не тому, *за что* его хвалят. Именно поэтому оно применимо там,
где функцию награды написать трудно, а показать правильное поведение — легко.

Откуда берётся эксперт в этой лаборатории
------------------------------------------
Скриптовым правилом, действующим **по тому же наблюдению**, что видит агент
(`labrl.envs.demos.corridor_expert`). Это не мелочь: если эксперт пользуется
знанием, недоступным агенту, имитация учится невозможному — политика
идеально повторяет эксперта на записях и разваливается в среде, потому что
нужного признака во входе просто нет.

Альтернатива — `DemonstrationRecorder` Unity и запись руками в редакторе.
Она поддержана (:func:`load_unity_demo`), но не является частью примера:
пример, который без ручной операции не воспроизводится, нарушает
требование 1.4 инструкции.

Формат хранения
---------------
``.npz`` с двумя массивами: ``obs`` формы ``(N, obs_dim)`` и ``action``
формы ``(N, num_branches)``. Плоский, без разбиения на эпизоды: ни BC,
ни GAIL границы эпизодов не используют — обоим нужны пары состояние-действие.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

#: Тип скриптового эксперта: батч наблюдений -> батч действий.
ExpertPolicy = Callable[[np.ndarray], np.ndarray]


@dataclass
class Demonstrations:
    """Пары «наблюдение → действие», снятые с эксперта.

    Attributes:
        obs: ``(N, obs_dim)``.
        action: ``(N, num_branches)`` — индексы дискретных веток.
        episodes: сколько эпизодов вошло в набор (для отчёта).
        success_rate: доля эпизодов эксперта, закончившихся успехом.
            Значение ниже единицы — повод посмотреть на эксперта раньше,
            чем на алгоритм: имитация не может быть лучше того, что имитирует.
    """

    obs: np.ndarray
    action: np.ndarray
    episodes: int = 0
    success_rate: float = 1.0

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[1])

    @property
    def num_branches(self) -> int:
        return int(self.action.shape[1])

    def action_histogram(self, branch: int = 0) -> np.ndarray:
        """Сколько раз встречается каждое действие ветки.

        Сильный перекос — не ошибка, а свойство задачи (в змеевидном коридоре
        ходов «на восток» и «на запад» вчетверо больше, чем «на север»),
        но знать о нём нужно: обучение с учителем на несбалансированных
        классах даёт политику, склонную к большинству.
        """
        return np.bincount(self.action[:, branch].astype(np.int64))

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Случайный мини-батч ``(obs, action)`` с возвращением."""
        index = rng.integers(0, len(self), size=batch_size)
        return self.obs[index], self.action[index]

    def split(self, holdout: float, rng: np.random.Generator) -> tuple["Demonstrations", "Demonstrations"]:
        """Делит набор на обучающую и отложенную части.

        Отложенная часть нужна, чтобы отличить «BC выучил эксперта»
        от «BC запомнил записи»: точность на обучающих парах у достаточно
        большой сети доходит до единицы в любом случае.
        """
        if not 0.0 < holdout < 1.0:
            raise ValueError(f"holdout должен быть в (0, 1), получено {holdout}")
        order = rng.permutation(len(self))
        cut = int(len(self) * (1.0 - holdout))
        train, test = order[:cut], order[cut:]
        return (
            Demonstrations(self.obs[train], self.action[train], self.episodes, self.success_rate),
            Demonstrations(self.obs[test], self.action[test], self.episodes, self.success_rate),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, obs=self.obs, action=self.action,
            episodes=np.array(self.episodes), success_rate=np.array(self.success_rate),
        )
        return p

    @staticmethod
    def load(path: str | Path) -> "Demonstrations":
        data = np.load(Path(path))
        return Demonstrations(
            obs=data["obs"].astype(np.float32),
            action=data["action"].astype(np.int64),
            episodes=int(data["episodes"]) if "episodes" in data else 0,
            success_rate=float(data["success_rate"]) if "success_rate" in data else 1.0,
        )

    def describe(self) -> str:
        return (f"демонстраций: {len(self)} пар из {self.episodes} эпизодов, "
                f"успех эксперта {self.success_rate:.0%}, "
                f"obs_dim {self.obs_dim}, веток {self.num_branches}")


def record_demonstrations(
    vec,  # noqa: ANN001 — labrl.envs.vec_unity_env.VecUnityEnv
    expert: ExpertPolicy,
    episodes: int,
    max_steps: int = 100_000,
) -> Demonstrations:
    """Прогоняет эксперта в среде и записывает пары «наблюдение → действие».

    Args:
        vec: векторизованная среда.
        expert: функция ``(N, obs_dim) -> (N, num_branches)``.
        episodes: сколько **завершившихся** эпизодов записать.
        max_steps: страховка от эксперта, который не доходит до конца.

    Returns:
        :class:`Demonstrations`.

    Записываются только шаги слотов, которые действительно получили действие
    (``active``): слот, только что завершивший эпизод, наблюдения не отдаёт,
    и дописывать за него нечего.

    Первый (прогревочный) эпизод каждого слота **не отбрасывается**, в отличие
    от оценки: эксперт детерминирован и не «разогревается», а лишние пары
    имитации только полезны.
    """
    obs_out: list[np.ndarray] = []
    action_out: list[np.ndarray] = []
    successes: list[bool] = []

    obs = vec.reset()
    active = np.ones(vec.num_envs, dtype=bool)

    for _ in range(max_steps):
        if len(successes) >= episodes:
            break

        flat = vec.flatten_obs(obs)
        actions = np.asarray(expert(flat), dtype=np.int64)

        for slot in np.flatnonzero(active):
            obs_out.append(flat[slot])
            action_out.append(actions[slot])

        result = vec.step(actions.astype(np.int32))
        for slot in np.flatnonzero(result.done):
            # Исход берётся у среды, а не выводится из награды: см.
            # labrl.eval.success. Обрыв по времени успехом не является.
            successes.append(bool(result.terminated[slot]))

        obs = result.obs
        active = result.active

    if not successes:
        raise RuntimeError(
            f"за {max_steps} шагов эксперт не завершил ни одного эпизода; "
            "проверьте правило эксперта и MaxStep агента"
        )

    return Demonstrations(
        obs=np.stack(obs_out).astype(np.float32),
        action=np.stack(action_out).astype(np.int64),
        episodes=len(successes),
        success_rate=float(np.mean(successes[:episodes])),
    )


def corridor_expert(obs: np.ndarray) -> np.ndarray:
    """Скриптовый эксперт среды `E10_Imitation`.

    Правило: **не возвращайся туда, откуда пришёл; иди в любую свободную
    клетку**. В коридоре без развилок этого достаточно, чтобы дойти до выхода
    кратчайшим путём.

    Эксперт пользуется **только наблюдением агента** (`ENV_SPEC.md` среды, §4):

    ==========  =========================================
    индексы     что это
    ==========  =========================================
    7, 8, 9, 10 стена ли на север / юг / восток / запад
    12          направление предыдущего хода / 3, −1/3 в начале
    ==========  =========================================

    Порядок «щупалец» совпадает с порядком действий, поэтому индекс свободного
    направления — это сразу индекс действия.

    Args:
        obs: ``(N, obs_dim)``, ``obs_dim >= 13``.

    Returns:
        ``(N, 1)`` — индексы действий.
    """
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] < 13:
        raise ValueError(
            f"эксперт коридора ждёт наблюдение (N, >=13), получено {obs.shape}"
        )

    walls = obs[:, 7:11] > 0.5                       # (N, 4): север, юг, восток, запад
    previous = np.rint(obs[:, 12] * 3.0).astype(np.int64)

    # Обратное направление к предыдущему ходу: 0<->1 (север/юг), 2<->3 (восток/запад).
    opposite = np.where(previous < 0, -1, previous ^ 1)

    actions = np.zeros((obs.shape[0], 1), dtype=np.int64)
    for i in range(obs.shape[0]):
        free = [d for d in range(4) if not walls[i, d]]
        # Возврат назад допускается, только если других вариантов нет —
        # в тупике, которого в этом коридоре быть не должно.
        forward = [d for d in free if d != opposite[i]]
        choices = forward or free
        if not choices:
            # Все четыре стороны — стена. Такого в коридоре не бывает;
            # действие 0 здесь означает «стоим», а не осмысленный ход.
            actions[i, 0] = 0
            continue
        # На старте (previous < 0) и на поворотах свободных направлений
        # два; берётся первое по порядку север → юг → восток → запад.
        # Порядок детерминирован: эксперт обязан быть воспроизводимым,
        # иначе два запуска дают разные демонстрации.
        actions[i, 0] = choices[0]
    return actions


def load_unity_demo(path: str | Path) -> Demonstrations:
    """Читает `.demo`-файл, записанный `DemonstrationRecorder` в Unity.

    Нужен тем, кто хочет показать поведение руками, а не скриптом. Формат
    бинарный (protobuf), и разбирать его самостоятельно смысла нет: читалка
    входит в установленный пакет `mlagents`. Это не нарушает запрет 16.5 —
    запрещены сторонние реализации **алгоритмов обучения**, а здесь читается
    файл.

    Args:
        path: путь к `.demo`.

    Returns:
        :class:`Demonstrations` — только дискретные действия.

    Raises:
        ValueError: в файле непрерывные действия (эта лаборатория использует
            демонстрации только для дискретных сред).
    """
    # Импорт локальный: mlagents.trainers тянет за собой весь тренер,
    # и платить за это при каждом импорте labrl.envs не нужно.
    from mlagents.trainers.demo_loader import load_demonstration

    behavior_spec, trajectories, _ = load_demonstration(str(Path(path)))
    if behavior_spec.action_spec.continuous_size > 0:
        raise ValueError(
            "в демонстрации непрерывные действия; имитационные методы "
            "лаборатории рассчитаны на дискретные (ENV_SPEC.md среды E10, §5)"
        )

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for step in trajectories:
        obs_rows.append(np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in step.obs]))
        action_rows.append(np.asarray(step.action.discrete, dtype=np.int64).ravel())

    return Demonstrations(
        obs=np.stack(obs_rows).astype(np.float32),
        action=np.stack(action_rows).astype(np.int64),
        episodes=0,
        success_rate=float("nan"),
    )


def concatenate(parts: Sequence[Demonstrations]) -> Demonstrations:
    """Склеивает несколько наборов демонстраций в один."""
    if not parts:
        raise ValueError("нечего склеивать: список наборов пуст")
    return Demonstrations(
        obs=np.concatenate([p.obs for p in parts]),
        action=np.concatenate([p.action for p in parts]),
        episodes=sum(p.episodes for p in parts),
        success_rate=float(np.mean([p.success_rate for p in parts])),
    )
