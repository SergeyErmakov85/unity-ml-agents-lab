"""Цикл обучения многорукого бандита в среде Unity.

Здесь — сбор исходов из K параллельных арен, расписание ε, логирование по схеме
раздела 11 и периодическая оценка жадной политики. Правил выбора руки и
обновления оценок здесь **нет**: они целиком в
:mod:`labrl.algos.bandits` (требование 8.7).

Особенность среды `E00_Bandit`: эпизод длится **один шаг**. Поэтому слот арены
чередуется: на одном шаге он действует и сразу получает терминал, на следующем
Unity начинает новый эпизод и слот помечен ``active = False``. Переход
записывается только для слотов, которые действительно нажали руку, — то есть
для тех, что вернули терминал.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from labrl.algos.bandits import Bandit, BanditBatch
from labrl.envs.vec_unity_env import StepResult, VecUnityEnv
from labrl.logging.tb_logger import TBLogger, Tags
from labrl.utils.schedules import Schedule


@dataclass
class BanditTrainConfig:
    """Параметры цикла обучения (не метода — метод настраивается `BanditConfig`)."""

    #: Сколько шагов сбора выполнить. Шаг = один вызов `env.step()`.
    total_steps: int = 4_000
    #: Как часто оценивать жадную политику, в шагах.
    eval_every_steps: int = 500
    #: Сколько эпизодов замерять при оценке. Награда бернуллиевская, поэтому
    #: выборка должна быть большой: 200 нажатий дают стандартную ошибку ~0.03.
    eval_episodes: int = 200
    #: Как часто писать `Perf/Steps Per Second`, в шагах.
    perf_every_steps: int = 200
    #: Коэффициент забывания в скользящей оценке распределения выбранных рук.
    #: Нужен только для тега `Policy/Entropy` (см. ниже).
    action_ema_decay: float = 0.99
    #: Порог награды, выше которого эпизод считается успешным. Для награды
    #: из {0, 1} это «нажатие принесло награду».
    success_threshold: float = 0.0


@dataclass
class BanditTrainResult:
    """Итог обучения."""

    algo: Bandit
    episode_returns: list[float] = field(default_factory=list)
    eval_history: list[tuple[int, float, float]] = field(default_factory=list)
    env_steps: int = 0
    pulls: int = 0
    wall_time: float = 0.0

    @property
    def last_eval_reward(self) -> float:
        return self.eval_history[-1][1] if self.eval_history else float("nan")

    @property
    def last_eval_success(self) -> float:
        return self.eval_history[-1][2] if self.eval_history else float("nan")


@dataclass
class EvalOutcome:
    """Результат оценки и наблюдение, на котором она остановилась."""

    mean_reward: float
    success_rate: float
    mean_length: float
    obs: list[np.ndarray]


def distribution_entropy(probabilities: np.ndarray) -> float:
    """Энтропия распределения, нат. Нулевые вероятности пропускаются."""
    p = np.asarray(probabilities, dtype=np.float64)
    total = p.sum()
    if total <= 0.0:
        return 0.0
    p = p[p > 0.0] / total
    return float(-(p * np.log(p)).sum())


def evaluate_greedy(
    vec: VecUnityEnv,
    algo: Bandit,
    obs: list[np.ndarray],
    episodes: int,
    success_threshold: float = 0.0,
    max_steps: int | None = None,
) -> EvalOutcome:
    """Оценка жадной политики — той же, что уйдёт в ONNX.

    Среда не сбрасывается: повторный ``env.reset()`` стоит шага с нулевым
    действием (T-7 в docs/07_TROUBLESHOOTING.md). Вместо сброса — прогрев:
    эпизод, начатый под разведочной политикой, доигрывается жадной
    и отбрасывается. Для бандита это ровно одно нажатие на слот.
    """
    budget = max_steps if max_steps is not None else 4 * episodes + 4 * vec.num_envs
    n = vec.num_envs

    running_return = np.zeros(n)
    counting = np.zeros(n, dtype=bool)
    returns: list[float] = []

    for _ in range(budget):
        if len(returns) >= episodes:
            break
        actions = algo.greedy_action(n)[:, None].astype(np.int32)
        result = vec.step(actions)
        running_return += result.reward

        for slot in np.flatnonzero(result.done):
            if counting[slot]:
                returns.append(float(running_return[slot]))
            else:
                counting[slot] = True  # прогревочный эпизод отброшен
            running_return[slot] = 0.0

        obs = result.obs

    if not returns:
        raise RuntimeError(
            f"за {budget} шагов оценки не завершился ни один эпизод после прогрева; "
            "проверьте MaxStep агента и условия завершения среды"
        )

    trimmed = np.array(returns[:episodes])
    return EvalOutcome(
        mean_reward=float(trimmed.mean()),
        success_rate=float((trimmed > success_threshold).mean()),
        # Эпизод бандита длится ровно один шаг — длина фиксирована определением среды.
        mean_length=1.0,
        obs=obs,
    )


def train_bandit(
    vec: VecUnityEnv,
    algo: Bandit,
    epsilon_schedule: Schedule,
    logger: TBLogger,
    cfg: BanditTrainConfig | None = None,
    seed: int = 0,
    on_eval: Callable[[int, float, float], None] | None = None,
    arm_probabilities: Sequence[float] | None = None,
) -> BanditTrainResult:
    """Обучает бандита в среде Unity.

    Args:
        vec: векторизованная среда (K арен = K параллельных бандитов
            с одинаковыми вероятностями рук).
        algo: алгоритм; его оценки меняются на месте.
        epsilon_schedule: расписание ε. Стратегии ``ucb`` и ``thompson``
            его игнорируют, но тег `Policy/Epsilon` пишется всегда — состав
            обязательных тегов не должен зависеть от метода.
        logger: логгер TensorBoard.
        cfg: параметры цикла.
        seed: сид генератора разведки.
        on_eval: колбэк ``(step, mean_reward, success_rate)`` после оценки.
        arm_probabilities: истинные вероятности рук из `ENV_SPEC.md`.
            Необязательны и используются только для диагностического тега
            `Custom/Regret Per Pull`: в реальной задаче они неизвестны.
    """
    cfg = cfg or BanditTrainConfig()
    rng = np.random.default_rng(seed)
    result = BanditTrainResult(algo=algo)

    obs = vec.reset()
    n = vec.num_envs

    # Скользящее распределение выбранных рук. Нужно для тега `Policy/Entropy`:
    # у бандита нет «распределения политики внутри сети», но есть наблюдаемая
    # частота выборов, и её энтропия — честная мера того, насколько широка
    # разведка. Для всех трёх стратегий она измеряется одинаково.
    action_ema = np.full(algo.num_actions, 1.0 / algo.num_actions)

    started = time.perf_counter()
    perf_mark, perf_step = started, 0
    metrics: dict[str, float] = {}

    for step in range(1, cfg.total_steps + 1):
        epsilon = float(epsilon_schedule(step))

        action = algo.act(n, epsilon, rng)
        acted = vec.step(action[:, None].astype(np.int32))

        batch = _build_batch(action, acted)
        metrics = algo.update(batch)
        result.pulls += int(batch.action.size)

        if batch.action.size:
            counts = np.bincount(batch.action, minlength=algo.num_actions).astype(np.float64)
            action_ema = cfg.action_ema_decay * action_ema + (1.0 - cfg.action_ema_decay) * counts / counts.sum()

        for slot in np.flatnonzero(acted.done):
            reward = float(acted.reward[slot])
            result.episode_returns.append(reward)
            logger.scalar(Tags.CUMULATIVE_REWARD, reward, step)
            logger.scalar(Tags.EPISODE_LENGTH, 1.0, step)

        # --- обязательные теги схемы 11.2 ---------------------------------
        # `Losses/Value Loss` для бандита — ошибка предсказания награды
        # |r − Q(a)|: это ровно то, что метод и уменьшает. Отдельного актора
        # у него нет, поэтому `Losses/Policy Loss` пишется нулём: состав тегов
        # не должен зависеть от алгоритма (docs/05_TENSORBOARD.md §2).
        logger.scalar(Tags.VALUE_LOSS, metrics["value_error_abs"], step)
        logger.scalar(Tags.POLICY_LOSS, 0.0, step)
        logger.scalar(Tags.ENTROPY, distribution_entropy(action_ema), step)
        # Скорость обучения инкрементального среднего — 1/N: она падает сама
        # по мере накопления опыта, и её график показывает, как быстро
        # «застывают» оценки.
        logger.scalar(Tags.LEARNING_RATE, metrics["mean_step_size"], step)
        logger.scalar(Tags.EPSILON, epsilon if algo.cfg.strategy == "eps_greedy" else 0.0, step)
        logger.custom("Value Error", metrics["value_error_abs"], step)
        logger.custom("Q Max", metrics["q_max"], step)
        logger.custom("Q Mean", metrics["q_mean"], step)
        logger.custom("Pulls", float(algo.pulls), step)
        if arm_probabilities is not None:
            logger.custom("Regret Per Pull", algo.regret_per_pull(np.asarray(arm_probabilities)), step)

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
            result.eval_history.append((step, outcome.mean_reward, outcome.success_rate))
            if on_eval is not None:
                on_eval(step, outcome.mean_reward, outcome.success_rate)
            obs = outcome.obs

    result.env_steps = cfg.total_steps
    result.wall_time = time.perf_counter() - started
    return result


def _build_batch(action: np.ndarray, result: StepResult) -> BanditBatch:
    """Собирает исходы шага.

    Учитываются только слоты, вернувшие терминал: эпизод бандита длится один
    шаг, поэтому «нажал руку» и «завершил эпизод» — одно и то же событие.
    Слот, который на этом шаге ничего не возвращал (Unity начинал в нём новый
    эпизод), нажатия не делал, и приписывать ему исход было бы выдумыванием
    данных.
    """
    pulled = result.done
    if not pulled.any():
        return BanditBatch(action=np.array([], dtype=np.int64), reward=np.array([]))
    return BanditBatch(action=action[pulled], reward=result.reward[pulled])
