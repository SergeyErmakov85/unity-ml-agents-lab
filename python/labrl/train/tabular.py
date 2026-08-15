"""Цикл обучения табличного Q-learning в среде Unity.

Что здесь происходит и чего здесь нет. Здесь — сбор опыта из K параллельных
арен, расписание ε, логирование по схеме раздела 11 и периодическая оценка
детерминированной политики. Здесь **нет** правила обновления: оно целиком
в :meth:`labrl.algos.tabular.q_learning.QLearning.update` (требование 8.7).

Ключевая тонкость сбора опыта. Векторизованная среда асинхронна: слот, эпизод
которого только что закончился, на следующем шаге может ещё не иметь нового
наблюдения (``active=False``). Переход записывается **только** для слотов,
которые действительно получили действие и вернули результат; наблюдение
завершённого эпизода берётся из ``final_obs``, а не из ``obs`` — в ``obs`` там
уже может лежать начало нового эпизода.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from labrl.algos.tabular.q_learning import QLearning, Transition
from labrl.envs.state_encoders import StateEncoder
from labrl.envs.vec_unity_env import StepResult, VecUnityEnv
from labrl.logging.tb_logger import TBLogger, Tags
from labrl.utils.schedules import Schedule


@dataclass
class TabularTrainConfig:
    """Параметры цикла обучения (не метода — метод настраивается `QLearningConfig`)."""

    #: Сколько шагов сбора опыта выполнить. Шаг = один вызов `env.step()`,
    #: то есть до K переходов сразу.
    total_steps: int = 20_000
    #: Как часто оценивать детерминированную политику, в шагах.
    eval_every_steps: int = 2_000
    #: Сколько эпизодов прогонять при оценке.
    eval_episodes: int = 20
    #: Как часто писать `Perf/Steps Per Second`, в шагах.
    perf_every_steps: int = 500
    #: Считать ли эпизод успешным. По умолчанию — положительная суммарная награда,
    #: что для GridWorld означает «дошёл до цели», а не «попал в ловушку».
    success_threshold: float = 0.0


@dataclass
class TabularTrainResult:
    """Итог обучения."""

    algo: QLearning
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    eval_history: list[tuple[int, float, float]] = field(default_factory=list)  # (step, reward, success)
    env_steps: int = 0
    transitions: int = 0
    wall_time: float = 0.0

    @property
    def last_eval_reward(self) -> float:
        return self.eval_history[-1][1] if self.eval_history else float("nan")

    @property
    def last_eval_success(self) -> float:
        return self.eval_history[-1][2] if self.eval_history else float("nan")


def states_from_obs(obs: list[np.ndarray]) -> np.ndarray:
    """One-hot наблюдение -> индекс состояния.

    Частный случай кодирования: среда отдаёт one-hot вектор, и ``argmax``
    и есть перевод в индекс. Общий случай — кодировщик алгоритма
    (:mod:`labrl.envs.state_encoders`): для непрерывного наблюдения это сетка
    дискретизации, и она обязана попасть ещё и в граф ONNX. Функция оставлена
    как самостоятельный помощник и как пояснение к простому случаю.
    """
    return np.argmax(obs[0], axis=1).astype(np.int64)


def epsilon_greedy_entropy(epsilon: float, num_actions: int) -> float:
    """Энтропия ε-жадной политики, нат.

    Тег ``Policy/Entropy`` обязателен схемой 11.2, а у табличного метода нет
    распределения политики «внутри сети». Но ε-жадная политика — вполне себе
    распределение: жадное действие получает ``1 − ε + ε/A``, каждое из
    остальных — ``ε/A``. Её энтропия и логируется: при ε=1 она равна log A
    (полная разведка), при ε=0 — нулю (чистая эксплуатация).
    """
    if num_actions < 2:
        return 0.0
    p_greedy = 1.0 - epsilon + epsilon / num_actions
    p_other = epsilon / num_actions
    terms = [p_greedy] + [p_other] * (num_actions - 1)
    return float(-sum(p * math.log(p) for p in terms if p > 0.0))


@dataclass
class EvalOutcome:
    """Результат оценки и наблюдение, на котором она остановилась."""

    mean_reward: float
    success_rate: float
    mean_length: float
    obs: list[np.ndarray]


def evaluate_greedy(
    vec: VecUnityEnv,
    algo: QLearning,
    obs: list[np.ndarray],
    episodes: int,
    success_threshold: float = 0.0,
    max_steps: int | None = None,
) -> EvalOutcome:
    """Оценка **детерминированной** (жадной) политики.

    Оценивается именно жадная политика — та же, что уйдёт в ONNX; ε-жадная
    систематически хуже и приёмочным числом быть не может.

    Среда **не сбрасывается**. Причина не в экономии: повторный ``env.reset()``
    в ML-Agents стоит одного шага с нулевым действием, из-за чего первый эпизод
    после сброса начинается не в стартовой клетке и оказывается короче
    остальных — измеренная награда выходит завышенной (T-7 в troubleshooting).
    Вместо сброса выполняется «прогрев»: эпизоды, начатые ещё под ε-жадной
    политикой, доигрываются жадной и **отбрасываются**, а замеряются только
    эпизоды, целиком прожитые жадной политикой.

    Args:
        vec: среда.
        algo: алгоритм; используется только его жадная политика.
        obs: текущее наблюдение — оценка продолжает существующий поток шагов.
        episodes: сколько завершённых эпизодов замерить.
        success_threshold: порог награды, выше которого эпизод считается успешным.
        max_steps: предохранитель от зависания.

    Returns:
        :class:`EvalOutcome`; поле ``obs`` возвращается вызывающему, чтобы цикл
        обучения продолжился с того же места без сброса.
    """
    budget = max_steps if max_steps is not None else 400 * episodes
    n = vec.num_envs

    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)
    # Слот начинает засчитываться только после того, как доиграл эпизод,
    # начатый под политикой сбора опыта.
    counting = np.zeros(n, dtype=bool)

    returns: list[float] = []
    lengths: list[int] = []

    for _ in range(budget):
        if len(returns) >= episodes:
            break
        actions = algo.greedy_action(algo.encoder.index(obs[0]))[:, None].astype(np.int32)
        result = vec.step(actions)

        running_return += result.reward
        running_length += (result.active | result.done).astype(np.int64)

        for slot in np.flatnonzero(result.done):
            if counting[slot]:
                returns.append(float(running_return[slot]))
                lengths.append(int(running_length[slot]))
            else:
                counting[slot] = True  # прогревочный эпизод отброшен
            running_return[slot] = 0.0
            running_length[slot] = 0

        obs = result.obs

    if not returns:
        raise RuntimeError(
            f"за {budget} шагов оценки не завершился ни один эпизод после прогрева; "
            "проверьте MaxStep агента и условия завершения среды"
        )

    trimmed = np.array(returns[:episodes])
    return EvalOutcome(
        mean_reward=float(np.mean(trimmed)),
        success_rate=float(np.mean(trimmed > success_threshold)),
        mean_length=float(np.mean(lengths[:episodes])),
        obs=obs,
    )


def train_q_learning(
    vec: VecUnityEnv,
    algo: QLearning,
    epsilon_schedule: Schedule,
    logger: TBLogger,
    cfg: TabularTrainConfig | None = None,
    seed: int = 0,
    on_eval: Callable[[int, float, float], None] | None = None,
    lr_schedule: Schedule | None = None,
) -> TabularTrainResult:
    """Обучает табличный Q-learning в среде Unity.

    Args:
        vec: векторизованная среда (K арен = K параллельных сред).
        algo: алгоритм; его таблица меняется на месте.
        epsilon_schedule: расписание ε от номера шага сбора.
        logger: логгер TensorBoard.
        cfg: параметры цикла.
        seed: сид генератора выбора случайных действий.
        lr_schedule: расписание α. ``None`` — постоянное значение из конфига
            метода. Убывающее α требуется условием сходимости Роббинса–Монро
            (см. :meth:`labrl.algos.tabular.q_learning.QLearning.set_learning_rate`).
        on_eval: колбэк ``(step, mean_reward, success_rate)`` после каждой оценки —
            для прогресс-вывода в ноутбуке.

    Returns:
        :class:`TabularTrainResult`.
    """
    cfg = cfg or TabularTrainConfig()
    rng = np.random.default_rng(seed)
    result = TabularTrainResult(algo=algo)

    obs = vec.reset()
    n = vec.num_envs
    num_actions = algo.num_actions

    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)

    started = time.perf_counter()
    perf_mark = started
    perf_step = 0

    for step in range(1, cfg.total_steps + 1):
        epsilon = float(epsilon_schedule(step))
        if lr_schedule is not None:
            algo.set_learning_rate(float(lr_schedule(step)))

        state = algo.encoder.index(obs[0])
        action = algo.act(state, epsilon, rng)
        # Активные слоты — те, что действительно ждут действия. Действия
        # неактивных слотов среда игнорирует, но записывать их переходы нельзя.
        acted = vec.step(action[:, None].astype(np.int32))

        batch = _build_batch(state, action, acted, algo.encoder)
        metrics = algo.update(batch)
        result.transitions += int(batch.state.size)

        running_return += acted.reward
        running_length += (acted.active | acted.done).astype(np.int64)

        for slot in np.flatnonzero(acted.done):
            result.episode_returns.append(float(running_return[slot]))
            result.episode_lengths.append(int(running_length[slot]))
            logger.scalar(Tags.CUMULATIVE_REWARD, running_return[slot], step)
            logger.scalar(Tags.EPISODE_LENGTH, running_length[slot], step)
            running_return[slot] = 0.0
            running_length[slot] = 0

        # Обязательные теги схемы 11.2 -----------------------------------
        # У табличного метода нет отдельного актора, поэтому Losses/Policy Loss
        # пишется нулём: состав тегов не должен зависеть от алгоритма, иначе
        # графики разных методов не лягут на одну ось (docs/05_TENSORBOARD.md §2).
        logger.scalar(Tags.VALUE_LOSS, metrics["td_error_abs"], step)
        logger.scalar(Tags.POLICY_LOSS, 0.0, step)
        logger.scalar(Tags.ENTROPY, epsilon_greedy_entropy(epsilon, num_actions), step)
        logger.scalar(Tags.LEARNING_RATE, algo.cfg.learning_rate, step)
        logger.scalar(Tags.EPSILON, epsilon, step)
        logger.custom("TD Error", metrics["td_error_abs"], step)
        logger.custom("Q Max", metrics["q_max"], step)
        logger.custom("Q Mean", metrics["q_mean"], step)

        if acted.info.get("env_stats"):
            logger.env_stats(acted.info["env_stats"], step)

        if step % cfg.perf_every_steps == 0:
            now = time.perf_counter()
            logger.scalar(Tags.STEPS_PER_SECOND, (step - perf_step) / max(now - perf_mark, 1e-9), step)
            perf_mark, perf_step = now, step

        obs = acted.obs

        if step % cfg.eval_every_steps == 0 or step == cfg.total_steps:
            outcome = evaluate_greedy(vec, algo, obs, cfg.eval_episodes, cfg.success_threshold)
            logger.scalar(Tags.EVAL_MEAN_REWARD, outcome.mean_reward, step)
            logger.scalar(Tags.EVAL_SUCCESS_RATE, outcome.success_rate, step)
            logger.custom("Eval Episode Length", outcome.mean_length, step)
            result.eval_history.append((step, outcome.mean_reward, outcome.success_rate))
            if on_eval is not None:
                on_eval(step, outcome.mean_reward, outcome.success_rate)

            # Оценка шагала той же средой, поэтому счётчики текущих эпизодов
            # сбора опыта уже не относятся к сбору: обнуляем их и продолжаем
            # с наблюдения, на котором оценка остановилась. Сбрасывать среду
            # нельзя — это стоило бы лишнего шага (T-7).
            obs = outcome.obs
            running_return[:] = 0.0
            running_length[:] = 0

    result.env_steps = cfg.total_steps
    result.wall_time = time.perf_counter() - started
    return result


def _build_batch(
    state: np.ndarray,
    action: np.ndarray,
    result: StepResult,
    encoder: "StateEncoder | None" = None,
) -> Transition:
    """Собирает батч переходов из результата шага векторизованной среды.

    Берутся только слоты, вернувшие результат: либо завершившие эпизод
    (тогда следующее состояние — из ``final_obs``), либо получившие новое
    наблюдение (``active``). Слот, который ничего не вернул, перехода не даёт —
    подставлять туда что-либо значило бы выдумывать данные.
    """
    done = result.terminated | result.truncated
    usable = done | result.active
    if not usable.any():
        empty_i = np.array([], dtype=np.int64)
        return Transition(empty_i, empty_i, np.array([]), empty_i, np.array([], dtype=bool),
                          np.array([], dtype=bool))

    next_obs = np.where(done[:, None], result.final_obs[0], result.obs[0])
    next_state = (
        np.argmax(next_obs, axis=1).astype(np.int64) if encoder is None else encoder.index(next_obs)
    )

    return Transition(
        state=state[usable],
        action=action[usable],
        reward=result.reward[usable],
        next_state=next_state[usable],
        terminated=result.terminated[usable],
        truncated=result.truncated[usable],
    )
