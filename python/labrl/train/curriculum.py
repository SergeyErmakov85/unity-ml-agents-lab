"""Учебный план: повышение сложности среды по мере роста доли успехов.

Зачем это отдельный механизм, когда есть формирование награды
--------------------------------------------------------------
Формирование награды (`E06_Hunter3D`, `E08_SoccerArena`) даёт плотный сигнал
там, где редкая награда недостижима. Но в лабиринте потенциал по расстоянию
до выхода **вреден**: он тянет агента к стене, за которой цель, а обходной
путь ведёт «от» цели и штрафуется. Плотного сигнала, безопасного для этой
задачи, попросту нет.

Учебный план решает ту же проблему с другой стороны: он не меняет награду,
он меняет **задачу**. Лабиринт 5 × 5 с двумя стенами случайная политика
проходит регулярно — сигнал есть с первого эпизода. Дальше сложность растёт,
и на каждом шаге агент решает задачу, лишь немного более трудную, чем та,
которую уже умеет.

Как устроено здесь
------------------
План — список **уроков**. У каждого урока своя сложность и свой порог
перевода. Переход вверх происходит, когда измеряемая величина держится выше
порога, а урок длится не меньше ``min_steps``.

Что измеряется — вопрос принципиальный. Награда для этого не годится:
она зависит от сложности (на большой сетке путь длиннее и штраф за время
больше), и один и тот же порог означал бы разные вещи на разных уроках.
Годится **доля успехов**: «дошёл до выхода» значит одно и то же на любой сетке.

Почему план не откатывается назад
---------------------------------
Соблазн «упала доля успехов — вернуть предыдущий урок» велик, и в этом
модуле он сознательно не реализован. Доля успехов падает сразу после каждого
повышения — это и есть ожидаемое поведение, а не признак беды. Автоматический
откат по такому сигналу даёт колебания между двумя уровнями и обучение,
которое никуда не идёт. Если план оказался слишком крутым, его правят
в конфиге, а не по ходу прогона.

Как проверять, что агент действительно растёт
---------------------------------------------
Доля успехов **на своём уровне** почти ничего не говорит: она по построению
держится около порога повышения. Настоящая проверка — оценка на
**фиксированной максимальной** сложности; она же критерий приёмки примера.
См. :func:`evaluate_at_difficulty`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Lesson:
    """Один урок учебного плана.

    Attributes:
        difficulty: сложность среды на этом уроке, ``[0, 1]``. Уходит в Unity
            через ``EnvironmentParametersChannel`` как параметр ``difficulty``.
        threshold: значение измеряемой величины, при котором план переходит
            к следующему уроку.
        min_steps: минимальная длительность урока в шагах сбора. Защита от
            перевода по одному удачному замеру: доля успехов, посчитанная
            по двум десяткам эпизодов, шумит заметно.
    """

    difficulty: float
    threshold: float
    min_steps: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(f"difficulty должна быть в [0, 1], получено {self.difficulty}")
        if self.min_steps < 0:
            raise ValueError(f"min_steps должен быть >= 0, получено {self.min_steps}")


@dataclass
class Curriculum:
    """Учебный план: последовательность уроков и правило перехода.

    Args:
        lessons: уроки по возрастанию сложности. Последний урок порога
            не имеет — с него некуда переходить.

    Пример::

        plan = Curriculum([
            Lesson(difficulty=0.0, threshold=0.7, min_steps=20_000),
            Lesson(difficulty=0.5, threshold=0.6, min_steps=30_000),
            Lesson(difficulty=1.0, threshold=1.0),
        ])
    """

    lessons: Sequence[Lesson]
    index: int = 0
    #: Шаг, на котором начался текущий урок.
    lesson_started_at: int = 0
    #: История переходов: ``(шаг, новый индекс урока, значение метрики)``.
    history: list[tuple[int, int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lessons:
            raise ValueError("учебный план обязан содержать хотя бы один урок")
        difficulties = [lesson.difficulty for lesson in self.lessons]
        if difficulties != sorted(difficulties):
            raise ValueError(
                f"уроки обязаны идти по возрастанию сложности, получено {difficulties}"
            )

    @property
    def current(self) -> Lesson:
        return self.lessons[self.index]

    @property
    def difficulty(self) -> float:
        """Сложность, которую нужно передать в среду прямо сейчас."""
        return self.current.difficulty

    @property
    def finished(self) -> bool:
        """Достигнут ли последний урок."""
        return self.index >= len(self.lessons) - 1

    def report(self, step: int, measure: float) -> bool:
        """Сообщает плану очередной замер измеряемой величины.

        Args:
            step: текущий шаг сбора опыта.
            measure: значение метрики (обычно доля успехов на текущем уроке).

        Returns:
            ``True``, если урок сменился — тогда вызывающий обязан передать
            новую :attr:`difficulty` в среду. ``False``, если ничего
            не изменилось.
        """
        if self.finished:
            return False
        if step - self.lesson_started_at < self.current.min_steps:
            return False
        if measure < self.current.threshold:
            return False

        self.index += 1
        self.lesson_started_at = step
        self.history.append((step, self.index, float(measure)))
        return True

    def describe(self) -> str:
        """Человекочитаемое состояние плана — для отчёта ноутбука."""
        lines = [f"урок {self.index + 1} из {len(self.lessons)}: "
                 f"difficulty = {self.difficulty:.2f}, порог = {self.current.threshold:.2f}"]
        for step, index, measure in self.history:
            lines.append(f"  шаг {step:>7}: переход на урок {index + 1} "
                         f"(difficulty {self.lessons[index].difficulty:.2f}) при метрике {measure:.2f}")
        return "\n".join(lines)


def evaluate_at_difficulty(
    vec,  # noqa: ANN001 — labrl.envs.vec_unity_env.VecUnityEnv
    algo,  # noqa: ANN001 — любой алгоритм с deterministic_action
    channels,  # noqa: ANN001 — labrl.envs.side_channels.LabSideChannels
    obs,  # noqa: ANN001 — текущее наблюдение
    difficulty: float,
    episodes: int,
    success_rule: str,
    success_threshold: float = 0.0,
):  # noqa: ANN201 — labrl.train.onpolicy.EvalOutcome
    """Оценка на **фиксированной** сложности, не зависящей от учебного плана.

    Зачем нужна отдельно от обычной оценки. Доля успехов на текущем уроке
    держится около порога перевода — она измеряет не силу агента, а настройку
    плана. Осмысленно сравнивать сиды и прогоны можно только на одной и той же
    задаче, поэтому приёмка идёт на максимальной сложности.

    Сложность применяется **на границе эпизода** (`MazeArea.ResetEpisode`
    читает параметр заново каждый эпизод), поэтому эпизоды, начатые до вызова,
    доигрываются на старой. Их и так отбрасывает прогрев внутри
    :func:`labrl.train.onpolicy.evaluate_deterministic`.

    Args:
        difficulty: сложность, на которой оценивать.
        episodes: сколько эпизодов зачесть.
        success_rule: правило успеха (:mod:`labrl.eval.success`).
        success_threshold: порог для правила ``reward_above``.

    Returns:
        :class:`labrl.train.onpolicy.EvalOutcome`. Сложность после вызова
        остаётся установленной: вернуть её обязан вызывающий, иначе обучение
        продолжится на оценочной, а не на учебной.
    """
    # Импорт внутри функции: labrl.train.onpolicy импортирует расписания,
    # и импорт на уровне модуля замкнул бы цикл через labrl.train.__init__.
    from labrl.train.onpolicy import evaluate_deterministic

    channels.set_difficulty(float(difficulty))
    return evaluate_deterministic(
        vec, algo, obs, episodes,
        success_threshold=success_threshold, success_rule=success_rule,
    )


def build_curriculum(spec: Sequence[dict]) -> Curriculum:
    """Собирает план из списка словарей конфига.

    Формат в YAML::

        curriculum:
          - {difficulty: 0.0, threshold: 0.70, min_steps: 20000}
          - {difficulty: 0.5, threshold: 0.60, min_steps: 30000}
          - {difficulty: 1.0, threshold: 1.00}
    """
    return Curriculum([Lesson(**lesson) for lesson in spec])
