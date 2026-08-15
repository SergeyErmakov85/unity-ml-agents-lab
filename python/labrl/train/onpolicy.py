"""Цикл обучения on-policy методов (A2C, PPO) в среде Unity.

Один цикл на оба метода: они отличаются **только** правилом обновления
(`labrl.algos.a2c.A2C.update` против `labrl.algos.ppo.PPO.update`), а сбор
опыта, расписание оценок и логирование у них общие. Дублировать цикл значило
бы получить два слегка разных сбора опыта и потом сравнивать методы по
разнице, которой в них нет.

Ритм цикла отличается от off-policy. DQN обновляется почти на каждом шаге,
беря батч из буфера; здесь опыт копится ``rollout_steps`` шагов, затем идёт
одно обновление, и роллаут **выбрасывается**: он собран старой политикой
и после шага градиента больше не описывает текущую.

Разведка. У ε-жадных методов разведка задаётся расписанием снаружи; здесь она
внутри политики — это σ гауссианы, обучаемый параметр. Поэтому тег
`Policy/Epsilon` пишется нулём (состав обязательных тегов не должен зависеть
от метода), а фактическая ширина разведки видна в `Policy/Entropy`
и `Custom/Log Std`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from labrl.buffers.rollout import RolloutBuffer
from labrl.envs.vec_unity_env import VecUnityEnv
from labrl.logging.tb_logger import TBLogger, Tags
from labrl.utils.schedules import Schedule


class OnPolicyAlgo(Protocol):
    """Минимальный протокол алгоритма, который умеет вести этот цикл."""

    cfg: object

    def act(self, obs: np.ndarray, rng: np.random.Generator): ...
    def set_learning_rate(self, learning_rate: float) -> None: ...
    def deterministic_action(self, obs: np.ndarray) -> np.ndarray: ...
    def value(self, obs: np.ndarray) -> np.ndarray: ...
    def update(self, batch) -> dict[str, float]: ...


@dataclass
class OnPolicyTrainConfig:
    """Параметры цикла обучения (не метода)."""

    #: Шагов сбора опыта. Шаг = один env.step() по всем аренам.
    total_steps: int = 100_000
    #: Сколько шагов копить перед обновлением. Батч обновления получается
    #: примерно ``rollout_steps × число арен`` переходов.
    rollout_steps: int = 128
    #: Как часто оценивать детерминированную политику, в шагах.
    eval_every_steps: int = 10_000
    #: Сколько эпизодов прогонять при оценке.
    eval_episodes: int = 20
    #: Как часто писать `Perf/Steps Per Second`.
    perf_every_steps: int = 1_000
    #: Порог награды, выше которого эпизод считается успешным.
    success_threshold: float = 0.0


@dataclass
class OnPolicyTrainResult:
    """Итог обучения."""

    algo: object
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
    #: Маска слотов, ждущих действия на следующем шаге. Нужна циклу обучения,
    #: чтобы после оценки записывать переходы только тех слотов, что реально
    #: получат действие.
    active: np.ndarray


def evaluate_deterministic(
    vec: VecUnityEnv,
    algo: OnPolicyAlgo,
    obs: list[np.ndarray],
    episodes: int,
    success_threshold: float = 0.0,
    max_steps: int | None = None,
) -> EvalOutcome:
    """Оценка детерминированной политики — той же, что уйдёт в ONNX.

    Детерминированное действие — среднее гауссианы, приведённое к диапазону
    Unity; это **в точности** выход `deterministic_continuous_actions`
    экспортированного графа. Стохастическая политика систематически хуже
    и приёмочным числом быть не может.

    Среда не сбрасывается: повторный ``env.reset()`` стоит шага с нулевым
    действием (T-7 в docs/07_TROUBLESHOOTING.md). Вместо сброса — прогрев:
    эпизоды, начатые под стохастической политикой, доигрываются
    детерминированной и отбрасываются.
    """
    budget = max_steps if max_steps is not None else 2000 * episodes
    n = vec.num_envs

    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)
    counting = np.zeros(n, dtype=bool)

    returns: list[float] = []
    lengths: list[int] = []
    active = np.ones(n, dtype=bool)

    for _ in range(budget):
        if len(returns) >= episodes:
            break
        actions = algo.deterministic_action(obs[0])
        result = vec.step(actions)
        active = result.active

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
        mean_reward=float(trimmed.mean()),
        success_rate=float((trimmed > success_threshold).mean()),
        mean_length=float(np.mean(lengths[:episodes])),
        obs=obs,
        active=active,
    )


def train_on_policy(
    vec: VecUnityEnv,
    algo: OnPolicyAlgo,
    logger: TBLogger,
    cfg: OnPolicyTrainConfig | None = None,
    seed: int = 0,
    on_eval: Callable[[int, float, float], None] | None = None,
    lr_schedule: "Schedule | None" = None,
) -> OnPolicyTrainResult:
    """Обучает A2C или PPO в среде Unity.

    Args:
        vec: векторизованная среда (K арен = K параллельных сред).
        algo: алгоритм; его сети меняются на месте.
        logger: логгер TensorBoard.
        cfg: параметры цикла.
        seed: сид генератора разведочного шума и перемешивания.
        on_eval: колбэк ``(step, mean_reward, success_rate)`` после оценки.
        lr_schedule: расписание шага обучения. ``None`` — постоянный шаг
            из конфига метода. Затухающий шаг защищает уже выученную политику
            от блуждания после того, как задача решена
            (см. :meth:`labrl.algos.a2c.A2C.set_learning_rate`).
    """
    cfg = cfg or OnPolicyTrainConfig()
    rng = np.random.default_rng(seed)
    result = OnPolicyTrainResult(algo=algo)

    obs = vec.reset()
    n = vec.num_envs
    buffer = RolloutBuffer(num_envs=n, gamma=algo.cfg.gamma, gae_lambda=algo.cfg.gae_lambda)

    # После reset() действия ждут все слоты.
    acting = np.ones(n, dtype=bool)

    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)

    started = time.perf_counter()
    perf_mark, perf_step = started, 0
    metrics: dict[str, float] = {}

    for step in range(1, cfg.total_steps + 1):
        if lr_schedule is not None:
            algo.set_learning_rate(float(lr_schedule(step)))

        current_obs = obs[0]
        out = algo.act(current_obs, rng)
        acted = vec.step(out.env_action)

        # Ценность последнего состояния оборванного эпизода. Считается одним
        # батчем: обрыв по времени требует бутстрэппинга, и без него метод
        # выучил бы, что нехватка времени равносильна провалу.
        bootstrap = np.zeros(n, dtype=np.float32)
        if acted.truncated.any():
            bootstrap[acted.truncated] = algo.value(acted.final_obs[0][acted.truncated])

        stored = 0
        for slot in np.flatnonzero(acting):
            buffer.add(
                slot=int(slot),
                obs=current_obs[slot],
                action=out.raw_action[slot],
                log_prob=float(out.log_prob[slot]),
                value=float(out.value[slot]),
                reward=float(acted.reward[slot]),
                terminated=bool(acted.terminated[slot]),
                truncated=bool(acted.truncated[slot]),
                bootstrap_value=float(bootstrap[slot]),
            )
            stored += 1
        result.transitions += stored

        running_return += acted.reward
        running_length += (acted.active | acted.done).astype(np.int64)

        for slot in np.flatnonzero(acted.done):
            result.episode_returns.append(float(running_return[slot]))
            result.episode_lengths.append(int(running_length[slot]))
            logger.scalar(Tags.CUMULATIVE_REWARD, running_return[slot], step)
            logger.scalar(Tags.EPISODE_LENGTH, running_length[slot], step)
            running_return[slot] = 0.0
            running_length[slot] = 0

        obs = acted.obs
        # Действие на следующем шаге получат только слоты, ждущие решения.
        acting = acted.active

        # --- обновление ---------------------------------------------------
        if step % cfg.rollout_steps == 0 and len(buffer) > 0:
            # Продолжение цепочки GAE для слотов, чей последний записанный шаг
            # не завершил эпизод. Для слота, закончившего эпизод, это значение
            # не используется (см. RolloutBuffer.compute).
            last_values = algo.value(obs[0])
            metrics = algo.update(buffer.compute(last_values))
            buffer.clear()
            result.updates += 1

        # --- обязательные теги схемы 11.2 ---------------------------------
        logger.scalar(Tags.VALUE_LOSS, metrics.get("value_loss", 0.0), step)
        logger.scalar(Tags.POLICY_LOSS, metrics.get("policy_loss", 0.0), step)
        logger.scalar(Tags.ENTROPY, metrics.get("entropy", 0.0), step)
        logger.scalar(Tags.LEARNING_RATE, algo.cfg.learning_rate, step)
        # Разведка on-policy метода живёт в σ политики, а не в ε.
        logger.scalar(Tags.EPSILON, 0.0, step)
        logger.custom("Explained Variance", metrics.get("explained_variance", 0.0), step)
        logger.custom("Approx KL", metrics.get("approx_kl", 0.0), step)
        logger.custom("Log Std", metrics.get("log_std", 0.0), step)
        logger.custom("Grad Norm", metrics.get("grad_norm", 0.0), step)
        if "clip_fraction" in metrics:
            logger.custom("Clip Fraction", metrics["clip_fraction"], step)

        if acted.info.get("env_stats"):
            logger.env_stats(acted.info["env_stats"], step)

        if step % cfg.perf_every_steps == 0:
            now = time.perf_counter()
            logger.scalar(Tags.STEPS_PER_SECOND, (step - perf_step) / max(now - perf_mark, 1e-9), step)
            perf_mark, perf_step = now, step

        if step % cfg.eval_every_steps == 0 or step == cfg.total_steps:
            outcome = evaluate_deterministic(vec, algo, obs, cfg.eval_episodes, cfg.success_threshold)
            logger.scalar(Tags.EVAL_MEAN_REWARD, outcome.mean_reward, step)
            logger.scalar(Tags.EVAL_SUCCESS_RATE, outcome.success_rate, step)
            logger.custom("Eval Episode Length", outcome.mean_length, step)
            result.eval_history.append((step, outcome.mean_reward, outcome.success_rate))
            if on_eval is not None:
                on_eval(step, outcome.mean_reward, outcome.success_rate)

            # Оценка шагала той же средой: незавершённые эпизоды сбора опыта
            # к сбору уже не относятся, а недособранный роллаут содержит
            # переходы, оборванные посередине оценкой.
            obs = outcome.obs
            acting = outcome.active
            buffer.clear()
            running_return[:] = 0.0
            running_length[:] = 0

    result.env_steps = cfg.total_steps
    result.wall_time = time.perf_counter() - started
    return result
