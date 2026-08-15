"""Цикл обучения DQN в среде Unity.

Здесь — сбор опыта, наполнение буфера, расписание ε, логирование по схеме
раздела 11 и периодическая оценка. Правило обновления параметров целиком
в :meth:`labrl.algos.dqn.DQN.update` (требование 8.7).

Особенность сбора при `DecisionPeriod > 1`. Агент запрашивает решение не каждый
шаг физики, поэтому на большинстве шагов часть слотов неактивна. Переход
записывается **только** для слотов, вернувших результат; наблюдение
завершившегося эпизода берётся из ``final_obs``, а не из ``obs`` — там уже
может лежать начало нового эпизода.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from labrl.algos.dqn import DQN
from labrl.buffers.replay import ReplayBuffer
from labrl.envs.vec_unity_env import StepResult, VecUnityEnv
from labrl.logging.tb_logger import TBLogger, Tags
from labrl.utils.schedules import Schedule


@dataclass
class DQNTrainConfig:
    """Параметры цикла обучения (не метода)."""

    #: Шагов сбора опыта. Шаг = один env.step() по всем аренам.
    total_steps: int = 200_000
    #: Ёмкость буфера воспроизведения, переходов.
    buffer_size: int = 200_000
    #: Сколько переходов накопить, прежде чем начать обучение. Обучение на почти
    #: пустом буфере — это обучение на сильно коррелированной выборке, ради
    #: избавления от которой буфер и заводился.
    learning_starts: int = 2_000
    #: Раз во сколько шагов сбора выполнять обновление.
    train_freq: int = 1
    #: Сколько обновлений выполнять за раз.
    gradient_steps: int = 1
    #: Как часто оценивать детерминированную политику, в шагах.
    eval_every_steps: int = 10_000
    #: Сколько эпизодов прогонять при оценке.
    eval_episodes: int = 20
    #: Как часто писать Perf/Steps Per Second.
    perf_every_steps: int = 1_000
    #: Порог награды, выше которого эпизод считается успешным.
    success_threshold: float = 0.0


@dataclass
class DQNTrainResult:
    """Итог обучения."""

    algo: DQN
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    eval_history: list[tuple[int, float, float]] = field(default_factory=list)
    env_steps: int = 0
    transitions: int = 0
    wall_time: float = 0.0

    @property
    def last_eval_reward(self) -> float:
        return self.eval_history[-1][1] if self.eval_history else float("nan")

    @property
    def last_eval_success(self) -> float:
        return self.eval_history[-1][2] if self.eval_history else float("nan")

    @property
    def best_eval_reward(self) -> float:
        return max((r for _, r, _ in self.eval_history), default=float("nan"))


@dataclass
class EvalOutcome:
    """Результат оценки и наблюдение, на котором она остановилась."""

    mean_reward: float
    success_rate: float
    mean_length: float
    obs: list[np.ndarray]


def epsilon_greedy_entropy(epsilon: float, num_actions: int) -> float:
    """Энтропия ε-жадной политики, нат. См. пояснение в `labrl.train.tabular`."""
    if num_actions < 2:
        return 0.0
    p_greedy = 1.0 - epsilon + epsilon / num_actions
    p_other = epsilon / num_actions
    terms = [p_greedy] + [p_other] * (num_actions - 1)
    return float(-sum(p * math.log(p) for p in terms if p > 0.0))


def evaluate_greedy(
    vec: VecUnityEnv,
    algo: DQN,
    obs: list[np.ndarray],
    episodes: int,
    success_threshold: float = 0.0,
    max_steps: int | None = None,
) -> EvalOutcome:
    """Оценка детерминированной (жадной) политики — той же, что уйдёт в ONNX.

    Среда не сбрасывается: повторный ``env.reset()`` стоит шага с нулевым
    действием и портит первый эпизод (T-7 в docs/07_TROUBLESHOOTING.md).
    Вместо сброса — прогрев: эпизоды, начатые под ε-жадной политикой,
    доигрываются жадной и отбрасываются.
    """
    budget = max_steps if max_steps is not None else 2000 * episodes
    n = vec.num_envs

    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)
    counting = np.zeros(n, dtype=bool)

    returns: list[float] = []
    lengths: list[int] = []

    for _ in range(budget):
        if len(returns) >= episodes:
            break
        actions = algo.greedy_action(obs[0])[:, None].astype(np.int32)
        result = vec.step(actions)

        running_return += result.reward
        running_length += (result.active | result.done).astype(np.int64)

        for slot in np.flatnonzero(result.done):
            if counting[slot]:
                returns.append(float(running_return[slot]))
                lengths.append(int(running_length[slot]))
            else:
                counting[slot] = True
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


def train_dqn(
    vec: VecUnityEnv,
    algo: DQN,
    epsilon_schedule: Schedule,
    logger: TBLogger,
    cfg: DQNTrainConfig | None = None,
    seed: int = 0,
    on_eval: Callable[[int, float, float], None] | None = None,
) -> DQNTrainResult:
    """Обучает DQN в среде Unity.

    Args:
        vec: векторизованная среда (K арен = K параллельных сред).
        algo: алгоритм; его сети меняются на месте.
        epsilon_schedule: расписание ε от номера шага сбора.
        logger: логгер TensorBoard.
        cfg: параметры цикла.
        seed: сид генераторов разведки и выборки из буфера.
        on_eval: колбэк ``(step, mean_reward, success_rate)`` после оценки.
    """
    cfg = cfg or DQNTrainConfig()
    rng = np.random.default_rng(seed)
    result = DQNTrainResult(algo=algo)

    obs = vec.reset()
    n = vec.num_envs
    obs_dim = obs[0].shape[1]
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, seed=seed)

    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)

    started = time.perf_counter()
    perf_mark, perf_step = started, 0
    last_metrics: dict[str, float] = {}

    for step in range(1, cfg.total_steps + 1):
        epsilon = float(epsilon_schedule(step))

        current_obs = obs[0]
        action = algo.act(current_obs, epsilon, rng)
        acted = vec.step(action[:, None].astype(np.int32))

        added = _store_transitions(buffer, current_obs, action, acted)
        result.transitions += added

        running_return += acted.reward
        running_length += (acted.active | acted.done).astype(np.int64)

        for slot in np.flatnonzero(acted.done):
            result.episode_returns.append(float(running_return[slot]))
            result.episode_lengths.append(int(running_length[slot]))
            logger.scalar(Tags.CUMULATIVE_REWARD, running_return[slot], step)
            logger.scalar(Tags.EPISODE_LENGTH, running_length[slot], step)
            running_return[slot] = 0.0
            running_length[slot] = 0

        # --- обновление ---------------------------------------------------
        if len(buffer) >= cfg.learning_starts and step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                last_metrics = algo.update(buffer.sample(algo.cfg.batch_size))

        # --- обязательные теги схемы 11.2 ---------------------------------
        # Losses/Policy Loss пишется нулём: у DQN нет отдельного актора, но
        # состав тегов не должен зависеть от алгоритма — иначе графики разных
        # методов не лягут на одну ось (docs/05_TENSORBOARD.md §2).
        logger.scalar(Tags.VALUE_LOSS, last_metrics.get("loss", 0.0), step)
        logger.scalar(Tags.POLICY_LOSS, 0.0, step)
        logger.scalar(Tags.ENTROPY, epsilon_greedy_entropy(epsilon, algo.num_actions), step)
        logger.scalar(Tags.LEARNING_RATE, algo.cfg.learning_rate, step)
        logger.scalar(Tags.EPSILON, epsilon, step)
        logger.custom("TD Error", last_metrics.get("td_error_abs", 0.0), step)
        logger.custom("Q Max", last_metrics.get("q_max", 0.0), step)
        logger.custom("Q Mean", last_metrics.get("q_mean", 0.0), step)
        logger.custom("Grad Norm", last_metrics.get("grad_norm", 0.0), step)
        logger.custom("Buffer Size", float(len(buffer)), step)

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

            obs = outcome.obs
            running_return[:] = 0.0
            running_length[:] = 0

    result.env_steps = cfg.total_steps
    result.wall_time = time.perf_counter() - started
    return result


def _store_transitions(
    buffer: ReplayBuffer,
    obs: np.ndarray,
    action: np.ndarray,
    result: StepResult,
) -> int:
    """Кладёт в буфер переходы тех слотов, что вернули результат."""
    done = result.terminated | result.truncated
    usable = done | result.active
    if not usable.any():
        return 0

    next_obs = np.where(done[:, None], result.final_obs[0], result.obs[0])

    return buffer.add_batch(
        obs=obs[usable],
        action=action[usable],
        reward=result.reward[usable],
        next_obs=next_obs[usable],
        terminated=result.terminated[usable],
        truncated=result.truncated[usable],
    )
