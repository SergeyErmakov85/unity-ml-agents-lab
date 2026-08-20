"""Цикл обучения с самоигрой: MA-POCA в `E08_SoccerArena`.

Почему обычный цикл здесь не годится
------------------------------------
Все предыдущие циклы лаборатории обучают политику в **неизменной** среде.
В футболе среда для обучаемой команды включает соперника, который тоже
обучается. Получается погоня за движущейся мишенью: политика подстраивается
под соперника, соперник — под неё, и обе могут бесконечно кружить, не
становясь сильнее. Хуже того, обычная кривая награды перестаёт что-либо
означать: ничья 0:0 против сильного соперника и ничья против слабого дают
одно и то же число.

Ответ урока 3.2 — **self-play с пулом снимков**:

1. Учится **одна** команда (здесь — West, `TeamId = 0`).
2. Соперником управляет **замороженная** копия политики из пула прошлых
   версий. На протяжении блока шагов соперник не меняется, то есть среда
   для обучаемой команды снова стационарна.
3. Раз в ``save_every_steps`` текущая политика кладётся в пул; пул ограничен
   окном, старые снимки вытесняются.
4. С вероятностью ``play_against_latest_ratio`` соперником становится
   **текущая** политика — иначе обучаемая команда научится бить только
   устаревшие версии себя.

Почему учится всегда одна и та же сторона. Наблюдение в этой среде дано
в командной системе координат: для East мир зеркалится (`ENV_SPEC.md`, §4).
Значит, вход политики за West и за East устроен одинаково, и умение,
выученное за одну сторону, переносится на другую без дообучения.

ELO
---
Награда за матч плохо годится в меру прогресса: она измеряет силу
**относительно текущего соперника**, а тот меняется. ELO измеряет силу
относительно всей истории соперников: каждому снимку в пуле приписан
рейтинг, и после матча рейтинги обновляются по стандартной формуле::

    E = 1 / (1 + 10^((R_соперника − R_наша)/400))
    R_наша ← R_наша + K·(S − E)

где ``S`` — 1 за победу, 0.5 за ничью, 0 за поражение. Растущий ELO при
стоящей на месте награде — нормальная и ожидаемая картина self-play.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from labrl.buffers.group_rollout import GroupRolloutBuffer
from labrl.envs.team_env import TeamStep, TeamUnityEnv
from labrl.logging.tb_logger import TBLogger, Tags
from labrl.utils.schedules import Schedule

#: Начальный рейтинг ELO. Значение условно: важны только разности рейтингов.
INITIAL_ELO = 1200.0


@dataclass
class SelfPlayConfig:
    """Параметры цикла обучения (не метода)."""

    #: Шагов сбора опыта. Шаг = один env.step() по всем аренам.
    total_steps: int = 200_000
    #: Сколько шагов копить перед обновлением.
    rollout_steps: int = 128
    #: Как часто менять соперника, в шагах. Слишком часто — среда снова
    #: нестационарна; слишком редко — политика переобучается под одного
    #: соперника.
    swap_every_steps: int = 5_000
    #: Как часто класть текущую политику в пул снимков.
    save_every_steps: int = 10_000
    #: Сколько снимков хранить. Старые вытесняются: играть против версии
    #: тысячедавней давности так же бесполезно, как против случайной.
    window: int = 10
    #: Доля матчей против **текущей** политики, а не против снимка.
    play_against_latest_ratio: float = 0.5
    #: Коэффициент K формулы ELO. Больше — быстрее реакция, больше шум.
    elo_k: float = 16.0
    #: Как часто оценивать, в шагах.
    eval_every_steps: int = 20_000
    #: Сколько матчей прогонять при оценке.
    eval_matches: int = 20
    #: Как часто писать `Perf/Steps Per Second`.
    perf_every_steps: int = 1_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.play_against_latest_ratio <= 1.0:
            raise ValueError(
                f"play_against_latest_ratio должен быть в [0, 1], "
                f"получено {self.play_against_latest_ratio}"
            )
        if self.window <= 0:
            raise ValueError(f"window должен быть > 0, получено {self.window}")


@dataclass
class SelfPlayResult:
    """Итог обучения."""

    algo: object
    match_returns: list[float] = field(default_factory=list)
    match_lengths: list[int] = field(default_factory=list)
    #: ``(шаг, средняя награда, доля побед)`` каждой оценки.
    eval_history: list[tuple[int, float, float]] = field(default_factory=list)
    elo_history: list[tuple[int, float]] = field(default_factory=list)
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
    def final_elo(self) -> float:
        return self.elo_history[-1][1] if self.elo_history else INITIAL_ELO


class OpponentPool:
    """Пул замороженных снимков политики с рейтингами ELO.

    Снимок — глубокая копия ``state_dict`` актора, а не сам модуль: копия
    модуля тянула бы за собой граф и оптимизатор, а нужны только веса.
    """

    def __init__(self, window: int, initial_elo: float = INITIAL_ELO) -> None:
        self.window = int(window)
        self.snapshots: list[dict[str, torch.Tensor]] = []
        self.ratings: list[float] = []
        self.initial_elo = float(initial_elo)

    def __len__(self) -> int:
        return len(self.snapshots)

    def push(self, policy: torch.nn.Module, rating: float) -> None:
        """Кладёт снимок текущей политики; вытесняет самый старый."""
        self.snapshots.append(copy.deepcopy(policy.state_dict()))
        self.ratings.append(float(rating))
        if len(self.snapshots) > self.window:
            self.snapshots.pop(0)
            self.ratings.pop(0)

    def sample(self, rng: np.random.Generator) -> int:
        """Индекс случайного снимка. Равномерно: взвешивание по рейтингу
        свело бы разнообразие соперников к одному самому сильному, а
        устойчивость к слабым и странным соперникам — часть задачи."""
        return int(rng.integers(len(self.snapshots)))


def elo_expected(rating: float, opponent_rating: float) -> float:
    """Ожидаемый результат по формуле ELO: доля очка в [0, 1]."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def elo_update(rating: float, opponent_rating: float, score: float, k: float) -> tuple[float, float]:
    """Новые рейтинги обоих после матча с результатом ``score`` (1/0.5/0)."""
    expected = elo_expected(rating, opponent_rating)
    delta = k * (score - expected)
    return rating + delta, opponent_rating - delta


def match_score(reward: float, tolerance: float = 1e-6) -> float:
    """Результат матча из награды команды: 1 победа, 0.5 ничья, 0 поражение.

    Награда команды за матч складывается из гола (±1 с учётом времени)
    и платы за время (−0.5 максимум), поэтому нулевую ничью от проигрыша
    отделяет знак: гол даёт вклад не меньше 0.5 по модулю, а плата за время
    в отсутствие гола — ровно −0.5 при полном эпизоде.
    """
    if reward > 0.5 - tolerance:
        return 1.0
    if reward < -0.5 - tolerance:
        return 0.0
    return 0.5


def _flat(step: TeamStep) -> np.ndarray:
    """``(G, n, obs_dim)`` -> ``(G·n, obs_dim)`` для прогона политики одним батчем."""
    groups, n, dim = step.obs.shape
    return step.obs.reshape(groups * n, dim)


def _act_with(policy: torch.nn.Module, obs: np.ndarray, device: torch.device,
              deterministic: bool = False) -> np.ndarray:
    """Действия произвольной политики для плоского батча наблюдений."""
    tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
    with torch.no_grad():
        if deterministic:
            action = policy.greedy(tensor)
        else:
            action, _ = policy.sample(tensor)
    return action.cpu().numpy().astype(np.int64)


def evaluate_selfplay(
    env: TeamUnityEnv,
    algo,  # noqa: ANN001 — MAPOCA, но протокол здесь избыточен
    opponent: torch.nn.Module,
    steps_out: dict[int, TeamStep],
    matches: int,
    learner_team: int,
    opponent_team: int,
    max_steps: int = 100_000,
) -> tuple[float, float, float, dict[int, TeamStep]]:
    """Прогоняет ``matches`` матчей детерминированной политикой обеих сторон.

    Returns:
        ``(средняя награда, доля побед, средняя длина, состояние среды)``.

    Оценка идёт **той же** средой, что и обучение: отдельного билда нет,
    и после оценки цикл обязан продолжить с возвращённого состояния,
    а недособранный роллаут — выбросить.
    """
    device = algo.device
    groups = env.num_groups
    n = env.team_size

    returns: list[float] = []
    lengths: list[int] = []
    running = np.zeros(groups)
    running_len = np.zeros(groups, dtype=np.int64)
    # Матчи, начатые до входа в оценку, не считаются: их часть сыграна
    # стохастической политикой сбора опыта.
    counting = np.zeros(groups, dtype=bool)

    steps = steps_out
    for _ in range(max_steps):
        if len(returns) >= matches:
            break

        learner = steps[learner_team]
        rival = steps[opponent_team]

        actions = {
            learner_team: _act_with(algo.policy_net, _flat(learner), device, deterministic=True)
            .reshape(groups, n, -1),
            opponent_team: _act_with(opponent, _flat(rival), device, deterministic=True)
            .reshape(groups, n, -1),
        }
        steps = env.step(actions)

        learner = steps[learner_team]
        running += learner.reward
        running_len += learner.awaiting.astype(np.int64)

        for group in np.flatnonzero(learner.done):
            if counting[group]:
                returns.append(float(running[group]))
                lengths.append(int(running_len[group]))
            else:
                counting[group] = True
            running[group] = 0.0
            running_len[group] = 0

    if not returns:
        raise RuntimeError(
            f"за {max_steps} шагов оценки не завершился ни один матч после прогрева; "
            "проверьте MaxStep агента и условия завершения среды"
        )

    trimmed = np.array(returns[:matches])
    wins = np.array([match_score(r) for r in trimmed])
    return float(trimmed.mean()), float((wins == 1.0).mean()), float(np.mean(lengths[:matches])), steps


def train_selfplay(
    env: TeamUnityEnv,
    algo,  # noqa: ANN001 — MAPOCA
    logger: TBLogger,
    cfg: SelfPlayConfig | None = None,
    seed: int = 0,
    learner_team: int = 0,
    opponent_team: int = 1,
    lr_schedule: Schedule | None = None,
    on_eval=None,  # noqa: ANN001 — Callable[[int, float, float], None]
) -> SelfPlayResult:
    """Обучает MA-POCA самоигрой.

    Args:
        env: среда двух команд.
        algo: :class:`labrl.algos.mapoca.MAPOCA`; его сети меняются на месте.
        logger: логгер TensorBoard.
        cfg: параметры цикла.
        seed: сид выбора соперника и перемешивания.
        learner_team: `TeamId` обучаемой команды.
        opponent_team: `TeamId` соперника.
        lr_schedule: расписание шага обучения (T-13).
        on_eval: колбэк ``(step, mean_reward, win_rate)`` после оценки.
    """
    cfg = cfg or SelfPlayConfig()
    rng = np.random.default_rng(seed)
    result = SelfPlayResult(algo=algo)

    groups = None
    steps = env.reset()
    groups = env.num_groups
    n = env.team_size

    buffer = GroupRolloutBuffer(
        num_groups=groups, gamma=algo.cfg.gamma, gae_lambda=algo.cfg.gae_lambda
    )

    # Соперник — отдельный модуль той же архитектуры. Копия, а не ссылка:
    # иначе «замороженный» соперник менялся бы вместе с обучаемой политикой,
    # и весь смысл пула снимков пропал бы.
    opponent = copy.deepcopy(algo.policy_net).to(algo.device).eval()
    for parameter in opponent.parameters():
        parameter.requires_grad_(False)

    pool = OpponentPool(window=cfg.window)
    pool.push(algo.policy_net, INITIAL_ELO)
    learner_elo = INITIAL_ELO
    opponent_index = 0
    opponent_is_latest = True

    running_return = np.zeros(groups)
    running_length = np.zeros(groups, dtype=np.int64)

    started = time.perf_counter()
    perf_mark, perf_step = started, 0
    metrics: dict[str, float] = {}

    def load_opponent() -> None:
        """Ставит соперником либо текущую политику, либо снимок из пула."""
        nonlocal opponent_index, opponent_is_latest
        if rng.random() < cfg.play_against_latest_ratio or len(pool) == 0:
            opponent.load_state_dict(algo.policy_net.state_dict())
            opponent_is_latest = True
        else:
            opponent_index = pool.sample(rng)
            opponent.load_state_dict(pool.snapshots[opponent_index])
            opponent_is_latest = False

    load_opponent()

    for step in range(1, cfg.total_steps + 1):
        if lr_schedule is not None:
            algo.set_learning_rate(float(lr_schedule(step)))

        learner_step = steps[learner_team]
        rival_step = steps[opponent_team]

        learner_obs = _flat(learner_step)
        learner_action, learner_log_prob = algo.act(learner_obs)
        learner_value = algo.team_value(learner_step.obs, learner_step.active.astype(np.float32))

        rival_action = _act_with(opponent, _flat(rival_step), algo.device)

        acted = env.step(
            {
                learner_team: learner_action.reshape(groups, n, -1),
                opponent_team: rival_action.reshape(groups, n, -1),
            }
        )
        learner_acted = acted[learner_team]

        # Ценность последнего состояния оборванного матча: обрыв по времени
        # требует бутстрэппинга, истинное завершение — нет (8.3).
        bootstrap = np.zeros(groups, dtype=np.float32)
        if learner_acted.truncated.any():
            final_values = algo.team_value(
                learner_acted.final_obs, learner_acted.final_active.astype(np.float32)
            )
            bootstrap[learner_acted.truncated] = final_values[learner_acted.truncated]

        for group in np.flatnonzero(learner_step.awaiting):
            buffer.add(
                group=int(group),
                obs=learner_step.obs[group],
                action=learner_action.reshape(groups, n, -1)[group],
                active=learner_step.active[group].astype(np.float32),
                log_prob=learner_log_prob.reshape(groups, n)[group],
                value=float(learner_value[group]),
                reward=float(learner_acted.reward[group]),
                terminated=bool(learner_acted.terminated[group]),
                truncated=bool(learner_acted.truncated[group]),
                bootstrap_value=float(bootstrap[group]),
            )

        running_return += learner_acted.reward
        running_length += learner_step.awaiting.astype(np.int64)

        for group in np.flatnonzero(learner_acted.done):
            reward = float(running_return[group])
            result.match_returns.append(reward)
            result.match_lengths.append(int(running_length[group]))
            logger.scalar(Tags.CUMULATIVE_REWARD, reward, step)
            logger.scalar(Tags.EPISODE_LENGTH, running_length[group], step)

            # ELO обновляется только против снимка: матч против самого себя
            # ничего не говорит об относительной силе.
            if not opponent_is_latest and pool.snapshots:
                score = match_score(reward)
                learner_elo, pool.ratings[opponent_index] = elo_update(
                    learner_elo, pool.ratings[opponent_index], score, cfg.elo_k
                )

            running_return[group] = 0.0
            running_length[group] = 0

        steps = acted

        # --- обновление ---------------------------------------------------
        if step % cfg.rollout_steps == 0 and len(buffer) > 0:
            last_values = algo.team_value(
                steps[learner_team].obs, steps[learner_team].active.astype(np.float32)
            )
            metrics = algo.update(buffer.compute(last_values))
            buffer.clear()
            result.updates += 1

        # --- пул снимков и смена соперника --------------------------------
        if step % cfg.save_every_steps == 0:
            pool.push(algo.policy_net, learner_elo)
        if step % cfg.swap_every_steps == 0:
            load_opponent()

        # --- обязательные теги схемы 11.2 ---------------------------------
        logger.scalar(Tags.VALUE_LOSS, metrics.get("value_loss", 0.0), step)
        logger.scalar(Tags.POLICY_LOSS, metrics.get("policy_loss", 0.0), step)
        logger.scalar(Tags.ENTROPY, metrics.get("entropy", 0.0), step)
        logger.scalar(Tags.LEARNING_RATE, algo.cfg.learning_rate, step)
        # Разведка здесь — энтропия дискретной политики, а не ε.
        logger.scalar(Tags.EPSILON, 0.0, step)
        logger.custom("Explained Variance", metrics.get("explained_variance", 0.0), step)
        logger.custom("Approx KL", metrics.get("approx_kl", 0.0), step)
        logger.custom("Grad Norm", metrics.get("grad_norm", 0.0), step)
        logger.custom("Baseline Loss", metrics.get("baseline_loss", 0.0), step)
        # Разброс базлайна по игрокам: ноль означает, что контрфактический
        # базлайн не различает вклад игроков и MA-POCA выродился в PPO.
        logger.custom("Baseline Spread", metrics.get("baseline_spread", 0.0), step)
        logger.custom("ELO", learner_elo, step)
        logger.custom("Opponent Pool Size", float(len(pool)), step)
        if "clip_fraction" in metrics:
            logger.custom("Clip Fraction", metrics["clip_fraction"], step)

        if step % cfg.perf_every_steps == 0:
            now = time.perf_counter()
            logger.scalar(Tags.STEPS_PER_SECOND, (step - perf_step) / max(now - perf_mark, 1e-9), step)
            perf_mark, perf_step = now, step

        if step % cfg.eval_every_steps == 0 or step == cfg.total_steps:
            # Оценка — против **последнего** снимка, а не против случайного:
            # иначе кривая оценки скакала бы от силы случайно выбранного
            # соперника, а не от прогресса политики.
            reference = copy.deepcopy(algo.policy_net).to(algo.device).eval()
            if pool.snapshots:
                reference.load_state_dict(pool.snapshots[-1])

            mean_reward, win_rate, mean_length, steps = evaluate_selfplay(
                env, algo, reference, steps, cfg.eval_matches, learner_team, opponent_team
            )
            logger.scalar(Tags.EVAL_MEAN_REWARD, mean_reward, step)
            logger.scalar(Tags.EVAL_SUCCESS_RATE, win_rate, step)
            logger.custom("Eval Episode Length", mean_length, step)
            result.eval_history.append((step, mean_reward, win_rate))
            result.elo_history.append((step, learner_elo))
            if on_eval is not None:
                on_eval(step, mean_reward, win_rate)

            # Оценка шагала той же средой: недособранный роллаут содержит
            # матчи, оборванные посередине оценкой.
            buffer.clear()
            running_return[:] = 0.0
            running_length[:] = 0

    result.env_steps = cfg.total_steps
    result.wall_time = time.perf_counter() - started
    if not result.elo_history:
        result.elo_history.append((cfg.total_steps, learner_elo))
    return result
