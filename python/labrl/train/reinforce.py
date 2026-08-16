"""Цикл обучения REINFORCE в среде Unity.

Отличие от всех предыдущих циклов лаборатории — единица обучения. DQN учится
на батче случайных переходов, PPO — на нарезке из N шагов, а REINFORCE —
на **завершённых эпизодах**: возврат ``G_t`` определён только тогда, когда
эпизод дожит до конца. Поэтому цикл копит траектории по слотам и обновляет
политику, накопив заданное число эпизодов.

Второе отличие — наблюдения. `E05_FoodCollector` отдаёт сетку ``(2, 8, 8)``
и вектор из 6 чисел, склеить их в один вектор нельзя, и цикл работает со
**списком** наблюдений, передавая его политике как есть.

Про обрыв по времени. Эпизод, оборванный по ``MaxStep``, не завершён:
возврат последнего шага обязан включать оценку того, что было бы дальше.
Без baseline такой оценки нет, и оборванный эпизод пришлось бы считать
как завершённый — это и есть скрытая цена «чистого» REINFORCE. С baseline
подставляется ``V(s_последнее)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from labrl.algos.reinforce import REINFORCE, EpisodeBatch, discounted_returns
from labrl.envs.vec_unity_env import VecUnityEnv
from labrl.eval.success import REWARD_ABOVE, check_rule, success_rate
from labrl.logging.tb_logger import TBLogger, Tags


@dataclass
class ReinforceTrainConfig:
    """Параметры цикла обучения (не метода)."""

    #: Шагов сбора опыта. Шаг = один env.step() по всем аренам.
    total_steps: int = 100_000
    #: Сколько завершённых эпизодов копить до обновления. Меньше — чаще
    #: обновления, но выше дисперсия оценки градиента.
    episodes_per_update: int = 16
    #: Как часто оценивать детерминированную политику, в шагах.
    eval_every_steps: int = 20_000
    #: Сколько эпизодов прогонять при оценке.
    eval_episodes: int = 16
    #: Как часто писать `Perf/Steps Per Second`.
    perf_every_steps: int = 1_000
    #: Порог награды для правила ``reward_above``.
    success_threshold: float = 0.0
    #: Как определять успех эпизода (:mod:`labrl.eval.success`).
    success_rule: str = REWARD_ABOVE


@dataclass
class ReinforceTrainResult:
    """Итог обучения."""

    algo: REINFORCE
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    eval_history: list[tuple[int, float, float]] = field(default_factory=list)
    env_steps: int = 0
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


class _Trajectory:
    """Незавершённая траектория одного слота."""

    __slots__ = ("obs", "continuous", "discrete", "rewards")

    def __init__(self, num_sensors: int) -> None:
        self.obs: list[list[np.ndarray]] = [[] for _ in range(num_sensors)]
        self.continuous: list[np.ndarray] = []
        self.discrete: list[np.ndarray] = []
        self.rewards: list[float] = []

    def add(self, obs: list[np.ndarray], slot: int, continuous: np.ndarray,
            discrete: np.ndarray, reward: float) -> None:
        for sensor, batch in enumerate(obs):
            self.obs[sensor].append(batch[slot].copy())
        self.continuous.append(continuous.copy())
        self.discrete.append(discrete.copy())
        self.rewards.append(float(reward))

    def __len__(self) -> int:
        return len(self.rewards)


def evaluate_deterministic(
    vec: VecUnityEnv,
    algo: REINFORCE,
    obs: list[np.ndarray],
    episodes: int,
    success_threshold: float = 0.0,
    max_steps: int | None = None,
    success_rule: str = REWARD_ABOVE,
) -> EvalOutcome:
    """Оценка детерминированной политики — той же, что уйдёт в ONNX."""
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
        result = vec.step(algo.deterministic_action(obs))
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


def train_reinforce(
    vec: VecUnityEnv,
    algo: REINFORCE,
    logger: TBLogger,
    cfg: ReinforceTrainConfig | None = None,
    seed: int = 0,
    on_eval: Callable[[int, float, float], None] | None = None,
    lr_schedule=None,
) -> ReinforceTrainResult:
    """Обучает REINFORCE в среде Unity.

    Args:
        vec: векторизованная среда (K арен = K параллельных сред).
        algo: алгоритм; его сети меняются на месте.
        logger: логгер TensorBoard.
        cfg: параметры цикла.
        seed: сид генератора разведки.
        on_eval: колбэк ``(step, mean_reward, success_rate)`` после оценки.
        lr_schedule: расписание шага обучения; ``None`` — постоянный шаг.
    """
    cfg = cfg or ReinforceTrainConfig()
    rng = np.random.default_rng(seed)
    result = ReinforceTrainResult(algo=algo)

    obs = vec.reset()
    n = vec.num_envs
    num_sensors = len(obs)

    trajectories = [_Trajectory(num_sensors) for _ in range(n)]
    ready: list[tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]] = []
    ready_episodes = 0

    acting = np.ones(n, dtype=bool)
    started = time.perf_counter()
    perf_mark, perf_step = started, 0
    metrics: dict[str, float] = {}

    for step in range(1, cfg.total_steps + 1):
        if lr_schedule is not None:
            algo.set_learning_rate(float(lr_schedule(step)))

        out = algo.act(obs, rng)
        acted = vec.step(out.env_action)

        for slot in np.flatnonzero(acting):
            trajectories[slot].add(obs, int(slot), out.raw_continuous[slot],
                                   out.discrete[slot], acted.reward[slot])

        # --- завершившиеся эпизоды ---------------------------------------
        finished = np.flatnonzero(acted.done)
        if finished.size:
            # Обрыв по времени бутстрэппится оценкой baseline последнего
            # состояния; истинное завершение — нулём.
            bootstrap = np.zeros(n, dtype=np.float32)
            if acted.truncated.any():
                final = [batch[acted.truncated] for batch in acted.final_obs]
                bootstrap[acted.truncated] = algo.baseline(final)

            for slot in finished:
                trajectory = trajectories[slot]
                if len(trajectory) == 0:
                    continue

                returns = discounted_returns(trajectory.rewards, algo.cfg.gamma,
                                             float(bootstrap[slot]))
                ready.append((
                    [np.stack(parts) for parts in trajectory.obs],
                    np.stack(trajectory.continuous),
                    np.stack(trajectory.discrete),
                    returns.astype(np.float32),
                ))
                ready_episodes += 1

                result.episode_returns.append(float(np.sum(trajectory.rewards)))
                result.episode_lengths.append(len(trajectory))
                logger.scalar(Tags.CUMULATIVE_REWARD, result.episode_returns[-1], step)
                logger.scalar(Tags.EPISODE_LENGTH, result.episode_lengths[-1], step)

                trajectories[slot] = _Trajectory(num_sensors)

        # --- обновление ---------------------------------------------------
        if ready_episodes >= cfg.episodes_per_update:
            batch = EpisodeBatch(
                obs=[np.concatenate([episode[0][sensor] for episode in ready])
                     for sensor in range(num_sensors)],
                continuous=np.concatenate([episode[1] for episode in ready]),
                discrete=np.concatenate([episode[2] for episode in ready]),
                returns=np.concatenate([episode[3] for episode in ready]),
            )
            metrics = algo.update(batch)
            result.updates += 1
            ready.clear()
            ready_episodes = 0

        # --- обязательные теги схемы 11.2 ---------------------------------
        logger.scalar(Tags.VALUE_LOSS, metrics.get("value_loss", 0.0), step)
        logger.scalar(Tags.POLICY_LOSS, metrics.get("policy_loss", 0.0), step)
        logger.scalar(Tags.ENTROPY, metrics.get("entropy", 0.0), step)
        logger.scalar(Tags.LEARNING_RATE, algo.cfg.learning_rate, step)
        # Разведка REINFORCE — стохастичность самой политики, а не ε.
        logger.scalar(Tags.EPSILON, 0.0, step)
        logger.custom("Return Mean", metrics.get("return_mean", 0.0), step)
        logger.custom("Return Std", metrics.get("return_std", 0.0), step)
        logger.custom("Log Std", metrics.get("log_std", 0.0), step)
        logger.custom("Grad Norm", metrics.get("grad_norm", 0.0), step)
        logger.custom("Episodes Pending", float(ready_episodes), step)

        if acted.info.get("env_stats"):
            logger.env_stats(acted.info["env_stats"], step)

        if step % cfg.perf_every_steps == 0:
            now = time.perf_counter()
            logger.scalar(Tags.STEPS_PER_SECOND, (step - perf_step) / max(now - perf_mark, 1e-9), step)
            perf_mark, perf_step = now, step

        obs = acted.obs
        acting = acted.active

        if step % cfg.eval_every_steps == 0 or step == cfg.total_steps:
            outcome = evaluate_deterministic(vec, algo, obs, cfg.eval_episodes,
                                             cfg.success_threshold, success_rule=cfg.success_rule)
            logger.scalar(Tags.EVAL_MEAN_REWARD, outcome.mean_reward, step)
            logger.scalar(Tags.EVAL_SUCCESS_RATE, outcome.success_rate, step)
            logger.custom("Eval Episode Length", outcome.mean_length, step)
            result.eval_history.append((step, outcome.mean_reward, outcome.success_rate))
            if on_eval is not None:
                on_eval(step, outcome.mean_reward, outcome.success_rate)

            # Оценка шагала той же средой: недособранные траектории оборваны
            # посередине, и досчитывать по ним возврат нельзя.
            obs = outcome.obs
            acting = outcome.active
            trajectories = [_Trajectory(num_sensors) for _ in range(n)]

    result.env_steps = cfg.total_steps
    result.wall_time = time.perf_counter() - started
    return result
