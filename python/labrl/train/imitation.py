"""Циклы имитационного обучения: BC и GAIL.

Два цикла, а не один, потому что они принципиально разные.

**BC** вообще не ходит в среду во время обучения: у него есть готовые пары
эксперта, и обучение — это перебор мини-батчей. Среда нужна только для того,
чтобы **измерить** результат: точность на парах и доля пройденных эпизодов —
разные величины, и первая систематически завышает вторую (сдвиг распределения,
см. :mod:`labrl.algos.bc`).

**GAIL** ходит в среду, как обычный on-policy метод, но награду берёт
не оттуда: её выдаёт дискриминатор. Отсюда единственное существенное отличие
от :mod:`labrl.train.onpolicy` — награда в буфер кладётся не та, что вернула
среда. Всё остальное (GAE, различение ``terminated``/``truncated``, оценка)
совпадает.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from labrl.buffers.rollout import RolloutBuffer
from labrl.envs.demos import Demonstrations
from labrl.eval.success import REWARD_ABOVE
from labrl.logging.tb_logger import TBLogger, Tags
from labrl.train.onpolicy import evaluate_deterministic
from labrl.utils.schedules import Schedule


@dataclass
class BCTrainConfig:
    """Параметры цикла BC (не метода)."""

    #: Сколько мини-батчей прогнать.
    steps: int = 5_000
    #: Как часто оценивать политику **в среде**, в шагах обучения.
    eval_every_steps: int = 500
    #: Сколько эпизодов прогонять при оценке.
    eval_episodes: int = 20
    #: Доля демонстраций, отложенная для проверки. Без неё «BC выучил
    #: эксперта» неотличимо от «BC запомнил записи».
    holdout: float = 0.2
    #: Как определять успех эпизода (:mod:`labrl.eval.success`).
    success_rule: str = REWARD_ABOVE
    #: Порог награды для правила ``reward_above``.
    success_threshold: float = 0.0


@dataclass
class ImitationResult:
    """Итог обучения имитацией."""

    algo: object
    #: ``(шаг, средняя награда, доля успехов)`` каждой оценки в среде.
    eval_history: list[tuple[int, float, float]] = field(default_factory=list)
    #: ``(шаг, точность на обучающих парах, точность на отложенных)``.
    accuracy_history: list[tuple[int, float, float]] = field(default_factory=list)
    steps: int = 0
    wall_time: float = 0.0

    @property
    def last_eval_reward(self) -> float:
        return self.eval_history[-1][1] if self.eval_history else float("nan")

    @property
    def last_eval_success(self) -> float:
        return self.eval_history[-1][2] if self.eval_history else float("nan")

    @property
    def last_holdout_accuracy(self) -> float:
        return self.accuracy_history[-1][2] if self.accuracy_history else float("nan")


def train_bc(
    demos: Demonstrations,
    algo,  # noqa: ANN001 — labrl.algos.bc.BC
    logger: TBLogger,
    vec=None,  # noqa: ANN001 — VecUnityEnv; None — обучать без оценки в среде
    cfg: BCTrainConfig | None = None,
    seed: int = 0,
    on_eval: Callable[[int, float, float], None] | None = None,
) -> ImitationResult:
    """Обучает политику на демонстрациях эксперта.

    Args:
        demos: демонстрации.
        algo: :class:`labrl.algos.bc.BC`.
        logger: логгер TensorBoard.
        vec: среда для оценки. ``None`` — оценивать только точность на парах.
        cfg: параметры цикла.
        seed: сид разбиения на обучающую и отложенную части.
        on_eval: колбэк ``(шаг, награда, доля успехов)``.

    Схема метрик 11.2 требует определённых тегов у любого прогона, но
    у обучения с учителем нет ни ценности, ни разведки. Соответствующие теги
    пишутся нулями — состав обязательных тегов не должен зависеть от метода,
    иначе сводка по прогонам перестанет быть сравнимой.
    """
    cfg = cfg or BCTrainConfig()
    rng = np.random.default_rng(seed)
    result = ImitationResult(algo=algo)

    train_demos, holdout_demos = demos.split(cfg.holdout, rng)
    started = time.perf_counter()
    obs = vec.reset() if vec is not None else None
    metrics: dict[str, float] = {}

    for step in range(1, cfg.steps + 1):
        metrics = algo.update(train_demos.sample(algo.cfg.batch_size, rng))

        logger.scalar(Tags.POLICY_LOSS, metrics["policy_loss"], step)
        logger.scalar(Tags.ENTROPY, metrics["entropy"], step)
        logger.scalar(Tags.LEARNING_RATE, algo.cfg.learning_rate, step)
        # У обучения с учителем нет ни критика, ни разведки.
        logger.scalar(Tags.VALUE_LOSS, 0.0, step)
        logger.scalar(Tags.EPSILON, 0.0, step)
        logger.custom("Batch Accuracy", metrics["accuracy"], step)
        logger.custom("Grad Norm", metrics["grad_norm"], step)

        if step % cfg.eval_every_steps == 0 or step == cfg.steps:
            train_accuracy = algo.accuracy(train_demos)
            holdout_accuracy = algo.accuracy(holdout_demos)
            result.accuracy_history.append((step, train_accuracy, holdout_accuracy))
            logger.custom("Train Accuracy", train_accuracy, step)
            logger.custom("Holdout Accuracy", holdout_accuracy, step)

            if vec is not None:
                outcome = evaluate_deterministic(
                    vec, algo, obs, cfg.eval_episodes,
                    success_threshold=cfg.success_threshold, success_rule=cfg.success_rule,
                )
                obs = outcome.obs
                logger.scalar(Tags.EVAL_MEAN_REWARD, outcome.mean_reward, step)
                logger.scalar(Tags.EVAL_SUCCESS_RATE, outcome.success_rate, step)
                logger.custom("Eval Episode Length", outcome.mean_length, step)
                # Обязательные теги схемы 11.2 берутся у ОЦЕНОЧНЫХ эпизодов:
                # обучение с учителем эпизодов не собирает вовсе, но эпизоды
                # оценки — настоящие, и оставлять теги пустыми значило бы
                # сделать прогон BC несравнимым с остальными.
                logger.scalar(Tags.CUMULATIVE_REWARD, outcome.mean_reward, step)
                logger.scalar(Tags.EPISODE_LENGTH, outcome.mean_length, step)
                result.eval_history.append((step, outcome.mean_reward, outcome.success_rate))
                if on_eval is not None:
                    on_eval(step, outcome.mean_reward, outcome.success_rate)

        if step % max(1, cfg.steps // 20) == 0:
            logger.scalar(Tags.STEPS_PER_SECOND,
                          step / max(time.perf_counter() - started, 1e-9), step)

    result.steps = cfg.steps
    result.wall_time = time.perf_counter() - started
    return result


@dataclass
class GAILTrainConfig:
    """Параметры цикла GAIL (не метода)."""

    #: Шагов сбора опыта. Шаг = один env.step() по всем аренам.
    total_steps: int = 100_000
    #: Сколько шагов копить перед обновлением политики и дискриминатора.
    rollout_steps: int = 256
    #: Как часто оценивать, в шагах.
    eval_every_steps: int = 10_000
    #: Сколько эпизодов прогонять при оценке.
    eval_episodes: int = 20
    #: Как часто писать `Perf/Steps Per Second`.
    perf_every_steps: int = 1_000
    #: Как определять успех эпизода (:mod:`labrl.eval.success`).
    success_rule: str = REWARD_ABOVE
    #: Порог награды для правила ``reward_above``.
    success_threshold: float = 0.0


def train_gail(
    vec,  # noqa: ANN001 — VecUnityEnv
    algo,  # noqa: ANN001 — labrl.algos.ppo_discrete.PPODiscrete
    gail,  # noqa: ANN001 — labrl.algos.gail.GAIL
    logger: TBLogger,
    cfg: GAILTrainConfig | None = None,
    seed: int = 0,
    on_eval: Callable[[int, float, float], None] | None = None,
    lr_schedule: Schedule | None = None,
) -> ImitationResult:
    """Обучает политику наградой дискриминатора.

    Отличие от :func:`labrl.train.onpolicy.train_on_policy` ровно одно:
    в буфер кладётся не та награда, которую вернула среда, а смесь награды
    дискриминатора и награды среды (:meth:`labrl.algos.gail.GAIL.mixed_reward`).
    При ``env_reward_weight = 0`` награда среды не используется вовсе — это
    чистая имитация, и она честно показывает, на что способен метод без
    функции награды.

    **Оценка при этом всегда идёт по награде СРЕДЫ.** Награда дискриминатора
    измеряет похожесть на эксперта, а не решение задачи; сравнивать по ней
    прогоны бессмысленно.
    """
    cfg = cfg or GAILTrainConfig()
    rng = np.random.default_rng(seed)
    result = ImitationResult(algo=algo)

    obs = vec.reset()
    n = vec.num_envs
    buffer = RolloutBuffer(num_envs=n, gamma=algo.cfg.gamma, gae_lambda=algo.cfg.gae_lambda)

    acting = np.ones(n, dtype=bool)
    running_return = np.zeros(n)
    running_length = np.zeros(n, dtype=np.int64)

    # Пары, собранные политикой, — обучающая выборка «не эксперта»
    # для дискриминатора. Копится тот же роллаут, что и для политики.
    policy_obs: list[np.ndarray] = []
    policy_action: list[np.ndarray] = []

    started = time.perf_counter()
    perf_mark, perf_step = started, 0
    metrics: dict[str, float] = {}
    disc_metrics: dict[str, float] = {}

    for step in range(1, cfg.total_steps + 1):
        if lr_schedule is not None:
            algo.set_learning_rate(float(lr_schedule(step)))

        current_obs = vec.flatten_obs(obs)
        out = algo.act(current_obs, rng)
        acted = vec.step(out.env_action)

        # Награда дискриминатора считается на паре (s, a), из которой
        # действовали, — не на следующем наблюдении.
        gail_reward = gail.reward(current_obs, out.env_action)
        reward = gail.mixed_reward(gail_reward, acted.reward)

        bootstrap = np.zeros(n, dtype=np.float32)
        if acted.truncated.any():
            bootstrap[acted.truncated] = algo.value(
                vec.flatten_obs(acted.final_obs)[acted.truncated])

        for slot in np.flatnonzero(acting):
            buffer.add(
                slot=int(slot),
                obs=current_obs[slot],
                action=out.raw_action[slot],
                log_prob=float(out.log_prob[slot]),
                value=float(out.value[slot]),
                reward=float(reward[slot]),
                terminated=bool(acted.terminated[slot]),
                truncated=bool(acted.truncated[slot]),
                bootstrap_value=float(bootstrap[slot]),
            )
            policy_obs.append(current_obs[slot])
            policy_action.append(out.env_action[slot])

        # Награда эпизода считается по СРЕДЕ: она измеряет решение задачи,
        # а награда дискриминатора — похожесть на эксперта.
        running_return += acted.reward
        running_length += (acted.active | acted.done).astype(np.int64)

        for slot in np.flatnonzero(acted.done):
            logger.scalar(Tags.CUMULATIVE_REWARD, running_return[slot], step)
            logger.scalar(Tags.EPISODE_LENGTH, running_length[slot], step)
            running_return[slot] = 0.0
            running_length[slot] = 0

        obs = acted.obs
        acting = acted.active

        # --- обновление ---------------------------------------------------
        if step % cfg.rollout_steps == 0 and len(buffer) > 0:
            last_values = algo.value(vec.flatten_obs(obs))
            metrics = algo.update(buffer.compute(last_values))
            disc_metrics = gail.update((np.stack(policy_obs), np.stack(policy_action)))
            buffer.clear()
            policy_obs.clear()
            policy_action.clear()

        # --- обязательные теги схемы 11.2 ---------------------------------
        logger.scalar(Tags.VALUE_LOSS, metrics.get("value_loss", 0.0), step)
        logger.scalar(Tags.POLICY_LOSS, metrics.get("policy_loss", 0.0), step)
        logger.scalar(Tags.ENTROPY, metrics.get("entropy", 0.0), step)
        logger.scalar(Tags.LEARNING_RATE, algo.cfg.learning_rate, step)
        logger.scalar(Tags.EPSILON, 0.0, step)
        logger.custom("Explained Variance", metrics.get("explained_variance", 0.0), step)
        logger.custom("Approx KL", metrics.get("approx_kl", 0.0), step)
        logger.custom("GAIL Reward", float(np.mean(gail_reward)), step)
        logger.custom("Discriminator Loss", disc_metrics.get("discriminator_loss", 0.0), step)
        # 0.5 означает «политика неотличима от эксперта» — цель метода;
        # 1.0 — «дискриминатор выиграл, сигнал политике пропал».
        logger.custom("Discriminator Accuracy",
                      disc_metrics.get("discriminator_accuracy", 0.0), step)
        logger.custom("Gradient Penalty", disc_metrics.get("gradient_penalty", 0.0), step)

        if acted.info.get("env_stats"):
            logger.env_stats(acted.info["env_stats"], step)

        if step % cfg.perf_every_steps == 0:
            now = time.perf_counter()
            logger.scalar(Tags.STEPS_PER_SECOND,
                          (step - perf_step) / max(now - perf_mark, 1e-9), step)
            perf_mark, perf_step = now, step

        if step % cfg.eval_every_steps == 0 or step == cfg.total_steps:
            outcome = evaluate_deterministic(
                vec, algo, obs, cfg.eval_episodes,
                success_threshold=cfg.success_threshold, success_rule=cfg.success_rule,
            )
            logger.scalar(Tags.EVAL_MEAN_REWARD, outcome.mean_reward, step)
            logger.scalar(Tags.EVAL_SUCCESS_RATE, outcome.success_rate, step)
            logger.custom("Eval Episode Length", outcome.mean_length, step)
            result.eval_history.append((step, outcome.mean_reward, outcome.success_rate))
            if on_eval is not None:
                on_eval(step, outcome.mean_reward, outcome.success_rate)

            # Оценка шагала той же средой: недособранный роллаут содержит
            # эпизоды, оборванные посередине оценкой.
            obs = outcome.obs
            acting = outcome.active
            buffer.clear()
            policy_obs.clear()
            policy_action.clear()
            running_return[:] = 0.0
            running_length[:] = 0

    result.steps = cfg.total_steps
    result.wall_time = time.perf_counter() - started
    return result
