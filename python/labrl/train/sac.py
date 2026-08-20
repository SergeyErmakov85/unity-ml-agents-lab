"""Цикл обучения SAC в среде Unity.

Здесь — сбор опыта, наполнение буфера воспроизведения, расписание обновлений,
логирование по схеме раздела 11 и периодическая оценка. Правило обновления
параметров целиком в :meth:`labrl.algos.sac.SAC.update` (требование 8.7).

Чем ритм отличается от on-policy цикла. PPO копит роллаут и выбрасывает его
после обновления; SAC хранит **весь** опыт и учится на нём многократно. Отсюда
и главное практическое отличие: обновлений на шаг среды здесь примерно столько
же, сколько шагов, — метод специально рассчитан на дорогие шаги симуляции.

Про разогрев. Первые ``learning_starts`` шагов действия берутся **равномерно
случайными**, а не из политики. Причина не в разведке: необученная политика
и так близка к случайной. Причина в том, что первые обновления на почти пустом
буфере учат критиков на сильно коррелированной выборке, и эта ошибка потом
долго вымывается.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from labrl.algos.sac import SAC
from labrl.buffers.replay import ReplayBuffer
from labrl.envs.vec_unity_env import StepResult, VecUnityEnv
from labrl.eval.success import REWARD_ABOVE, check_rule, success_rate
from labrl.logging.tb_logger import TBLogger, Tags


@dataclass
class SACTrainConfig:
    """Параметры цикла обучения (не метода)."""

    #: Шагов сбора опыта. Шаг = один env.step() по всем аренам.
    total_steps: int = 100_000
    #: Ёмкость буфера воспроизведения, переходов.
    buffer_size: int = 300_000
    #: Сколько переходов накопить случайной политикой, прежде чем учиться.
    learning_starts: int = 5_000
    #: Раз во сколько шагов сбора выполнять обновления.
    train_freq: int = 1
    #: Сколько обновлений выполнять за раз.
    gradient_steps: int = 1
    #: Как часто оценивать детерминированную политику, в шагах.
    eval_every_steps: int = 20_000
    #: Сколько эпизодов прогонять при оценке.
    eval_episodes: int = 20
    #: Как часто писать `Perf/Steps Per Second`.
    perf_every_steps: int = 1_000
    #: Порог награды для правила ``reward_above``.
    success_threshold: float = 0.0
    #: Как определять успех эпизода (:mod:`labrl.eval.success`).
    success_rule: str = REWARD_ABOVE


@dataclass
class SACTrainResult:
    """Итог обучения."""

    algo: SAC
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    eval_history: list[tuple[int, float, float]] = field(default_factory=list)
    env_steps: int = 0
    transitions: int = 0
    updates: int = 0
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
    active: np.ndarray


def evaluate_deterministic(
    vec: VecUnityEnv,
    algo: SAC,
    obs: list[np.ndarray],
    episodes: int,
    success_threshold: float = 0.0,
    max_steps: int | None = None,
    success_rule: str = REWARD_ABOVE,
) -> EvalOutcome:
    """Оценка детерминированной политики ``tanh(μ)`` — той же, что уйдёт в ONNX.

    Среда не сбрасывается: повторный ``env.reset()`` стоит шага с нулевым
    действием (T-7 в docs/07_TROUBLESHOOTING.md). Вместо сброса — прогрев:
    эпизоды, начатые под стохастической политикой, доигрываются
    детерминированной и отбрасываются.
    """
    check_rule(success_rule)
    budget = max_steps if max_steps is not None else 2000 * episodes
    n = vec.num_envs

    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)
    counting = np.zeros(n, dtype=bool)

    returns: list[float] = []
    lengths: list[int] = []
    terminated: list[bool] = []
    active = np.ones(n, dtype=bool)

    for _ in range(budget):
        if len(returns) >= episodes:
            break
        result = vec.step(algo.deterministic_action(vec.flatten_obs(obs)))
        active = result.active

        running_return += result.reward
        running_length += (result.active | result.done).astype(np.int64)

        for slot in np.flatnonzero(result.done):
            if counting[slot]:
                returns.append(float(running_return[slot]))
                lengths.append(int(running_length[slot]))
                terminated.append(bool(result.terminated[slot]))
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
        mean_reward=float(trimmed.mean()),
        success_rate=success_rate(success_rule, trimmed,
                                  np.array(terminated[:episodes]), success_threshold),
        mean_length=float(np.mean(lengths[:episodes])),
        obs=obs,
        active=active,
    )


def train_sac(
    vec: VecUnityEnv,
    algo: SAC,
    logger: TBLogger,
    cfg: SACTrainConfig | None = None,
    seed: int = 0,
    on_eval: Callable[[int, float, float], None] | None = None,
    lr_schedule=None,
) -> SACTrainResult:
    """Обучает SAC в среде Unity.

    Args:
        vec: векторизованная среда (K арен = K параллельных сред).
        algo: алгоритм; его сети меняются на месте.
        logger: логгер TensorBoard.
        cfg: параметры цикла.
        seed: сид генераторов разведки и выборки из буфера.
        on_eval: колбэк ``(step, mean_reward, success_rate)`` после оценки.
        lr_schedule: расписание шага обучения; ``None`` — постоянный шаг.
    """
    cfg = cfg or SACTrainConfig()
    rng = np.random.default_rng(seed)
    result = SACTrainResult(algo=algo)

    obs = vec.reset()
    n = vec.num_envs
    obs_dim = vec.flatten_obs(obs).shape[1]
    action_dim = algo.action_dim
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, seed=seed, action_dim=action_dim)

    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)

    started = time.perf_counter()
    perf_mark, perf_step = started, 0
    metrics: dict[str, float] = {}

    for step in range(1, cfg.total_steps + 1):
        if lr_schedule is not None:
            algo.set_learning_rate(float(lr_schedule(step)))

        current_obs = vec.flatten_obs(obs)
        if len(buffer) < cfg.learning_starts:
            # Разогрев: равномерно случайные действия во всём допустимом
            # диапазоне. Политика на этом этапе всё равно ничего не знает,
            # а равномерная выборка покрывает пространство лучше, чем
            # сэмплы из необученной гауссианы.
            action = rng.uniform(-1.0, 1.0, size=(n, action_dim)).astype(np.float32)
        else:
            action = algo.act(current_obs, rng)

        acted = vec.step(action)
        result.transitions += _store(buffer, current_obs, action, acted, vec.flatten_obs)

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
                metrics = algo.update(buffer.sample(algo.cfg.batch_size))
                result.updates += 1

        # --- обязательные теги схемы 11.2 ---------------------------------
        logger.scalar(Tags.VALUE_LOSS, metrics.get("value_loss", 0.0), step)
        logger.scalar(Tags.POLICY_LOSS, metrics.get("policy_loss", 0.0), step)
        logger.scalar(Tags.ENTROPY, metrics.get("entropy", 0.0), step)
        logger.scalar(Tags.LEARNING_RATE, algo.cfg.learning_rate, step)
        # Разведка SAC живёт в энтропийной части цели, а не в ε.
        logger.scalar(Tags.EPSILON, 0.0, step)
        logger.custom("Alpha", metrics.get("alpha", 0.0), step)
        logger.custom("Alpha Loss", metrics.get("alpha_loss", 0.0), step)
        logger.custom("TD Error", metrics.get("td_error_abs", 0.0), step)
        logger.custom("Q Mean", metrics.get("q_mean", 0.0), step)
        logger.custom("Q Max", metrics.get("q_max", 0.0), step)
        logger.custom("Grad Norm", metrics.get("grad_norm", 0.0), step)
        logger.custom("Buffer Size", float(len(buffer)), step)

        if acted.info.get("env_stats"):
            logger.env_stats(acted.info["env_stats"], step)

        if step % cfg.perf_every_steps == 0:
            now = time.perf_counter()
            logger.scalar(Tags.STEPS_PER_SECOND, (step - perf_step) / max(now - perf_mark, 1e-9), step)
            perf_mark, perf_step = now, step

        obs = acted.obs

        if step % cfg.eval_every_steps == 0 or step == cfg.total_steps:
            outcome = evaluate_deterministic(vec, algo, obs, cfg.eval_episodes,
                                             cfg.success_threshold, success_rule=cfg.success_rule)
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


def _store(
    buffer: ReplayBuffer,
    obs: np.ndarray,
    action: np.ndarray,
    result: StepResult,
    flatten: Callable[[list[np.ndarray]], np.ndarray],
) -> int:
    """Кладёт в буфер переходы тех слотов, что вернули результат."""
    done = result.terminated | result.truncated
    usable = done | result.active
    if not usable.any():
        return 0

    # Наблюдение завершившегося эпизода берётся из final_obs: в obs там уже
    # может лежать начало нового эпизода.
    next_obs = np.where(done[:, None], flatten(result.final_obs), flatten(result.obs))

    return buffer.add_batch(
        obs=obs[usable],
        action=action[usable],
        reward=result.reward[usable],
        next_obs=next_obs[usable],
        terminated=result.terminated[usable],
        truncated=result.truncated[usable],
    )
