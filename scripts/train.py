"""Запуск обучения без ноутбука — для CI и прогонов по нескольким сидам.

Использование::

    python scripts/train.py --config configs/E01_GridWorld__qlearning.yaml --seed 0
    python scripts/train.py --config configs/E01_GridWorld__qlearning.yaml --seed 0 --quick
    python scripts/train.py --config configs/E01_GridWorld__qlearning.yaml --all-seeds

Скрипт **не содержит** ни математики метода, ни цикла обучения: и то и другое
живёт в `labrl` (`labrl.algos.*` и `labrl.train.*`). Здесь только разбор конфига,
подключение среды, каталог прогона и экспорт результата — то же, что делает
ноутбук, тем же кодом. Так учебный и рабочий путь не могут разойтись.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from labrl.algos.a2c import A2C, A2CConfig  # noqa: E402
from labrl.algos.bandits import STRATEGIES, Bandit, BanditConfig  # noqa: E402
from labrl.algos.dqn import DQN, DQNConfig  # noqa: E402
from labrl.algos.ppo import PPO, PPOConfig  # noqa: E402
from labrl.algos.tabular.q_learning import QLearning, QLearningConfig  # noqa: E402
from labrl.envs.state_encoders import BoxDiscretizer, OneHotStateEncoder  # noqa: E402
from labrl.envs.unity_env import open_unity_env  # noqa: E402
from labrl.envs.vec_unity_env import VecUnityEnv  # noqa: E402
from labrl.export.onnx_export import ActionSpecLite, export_policy_to_onnx  # noqa: E402
from labrl.export.onnx_verify import verify_onnx_model  # noqa: E402
from labrl.logging.run_dir import create_run_dir  # noqa: E402
from labrl.logging.tb_logger import TBLogger, git_commit_hash  # noqa: E402
from labrl.nets.gaussian_policy import GaussianPolicyNetwork  # noqa: E402
from labrl.nets.mlp import MLPQNetwork, MLPValueNetwork  # noqa: E402
from labrl.train.bandit import BanditTrainConfig, train_bandit  # noqa: E402
from labrl.train.dqn import DQNTrainConfig, train_dqn  # noqa: E402
from labrl.train.onpolicy import OnPolicyTrainConfig, train_on_policy  # noqa: E402
from labrl.train.tabular import TabularTrainConfig, train_q_learning  # noqa: E402
from labrl.utils.checkpoint import save_checkpoint  # noqa: E402
from labrl.utils.config import ExperimentConfig, dump_config, load_config, resolve_path  # noqa: E402
from labrl.utils.schedules import ConstantSchedule, Schedule, build_schedule  # noqa: E402
from labrl.utils.seeding import resolve_device, set_global_seed  # noqa: E402

#: Во сколько раз сокращается бюджет шагов в режиме --quick.
QUICK_STEP_DIVISOR = 10
#: Минимальный бюджет быстрой проверки, чтобы прогон оставался осмысленным.
QUICK_MIN_STEPS = 2_000


def build_encoder(cfg: ExperimentConfig):
    """Создаёт кодировщик наблюдений для табличного метода.

    ``type: table`` — среда отдаёт one-hot вектор состояния (`E01`).
    ``type: discretized`` — наблюдение непрерывно и режется сеткой (`E02`);
    та же сетка уедет в граф ONNX, иначе Unity истолкует вход иначе, чем
    обучение (требование 10.7).
    """
    spec = cfg.network
    kind = spec.get("type", "table")
    if kind == "discretized":
        return BoxDiscretizer.uniform(spec["lows"], spec["highs"], spec["bins"])
    return OneHotStateEncoder(int(spec["num_states"]))


def learning_rate_schedule(cfg: ExperimentConfig) -> Schedule:
    """Расписание шага обучения.

    ``algo.learning_rate`` может быть числом (постоянное α) или блоком
    расписания, как ``algo.epsilon``. Постоянное α нарушает условие сходимости
    Роббинса–Монро и на задачах управления даёт колеблющуюся политику
    (docs/07_TROUBLESHOOTING.md, T-11), поэтому число здесь — сознательный выбор
    пользователя, а не значение по умолчанию.
    """
    spec = cfg.algo["learning_rate"]
    if isinstance(spec, dict):
        return build_schedule(spec)
    return ConstantSchedule(float(spec))


def build_algo(cfg: ExperimentConfig, obs_dim: int, num_actions: int, seed: int = 0):
    """Создаёт алгоритм по конфигу.

    Сеть собирается здесь же, из блока `network`. В ноутбуке этот шаг делает
    сам пользователь — алгоритм принимает любой nn.Module, удовлетворяющий
    протоколу (требование 8.4), и об архитектуре ничего не знает.
    """
    if cfg.algo_name in STRATEGIES:
        return Bandit(
            num_actions=num_actions,
            cfg=BanditConfig(
                strategy=cfg.algo["strategy"],
                ucb_c=float(cfg.algo.get("ucb_c", 1.414)),
                prior_alpha=float(cfg.algo.get("prior_alpha", 1.0)),
                prior_beta=float(cfg.algo.get("prior_beta", 1.0)),
                initial_value=float(cfg.algo.get("initial_value", 0.0)),
            ),
        )

    if cfg.algo_name == "qlearning":
        encoder = build_encoder(cfg)
        return QLearning(
            num_states=encoder.num_states,
            num_actions=int(cfg.network["num_actions"]),
            cfg=QLearningConfig(
                gamma=float(cfg.algo["gamma"]),
                learning_rate=float(learning_rate_schedule(cfg)(0)),
                initial_q=float(cfg.algo.get("initial_q", 0.0)),
            ),
            encoder=encoder,
        )

    if cfg.algo_name == "dqn":
        hidden = tuple(cfg.network.get("hidden_sizes", (128, 128)))
        activation = cfg.network.get("activation", "relu")
        device = resolve_device("auto")
        q_net = MLPQNetwork(obs_dim, num_actions, hidden, activation)
        target_net = MLPQNetwork(obs_dim, num_actions, hidden, activation)
        return DQN(
            q_net, target_net,
            DQNConfig(
                gamma=float(cfg.algo["gamma"]),
                learning_rate=float(cfg.algo["learning_rate"]),
                batch_size=int(cfg.algo["batch_size"]),
                target_update_interval=int(cfg.algo["target_update_interval"]),
                max_grad_norm=float(cfg.algo.get("max_grad_norm", 10.0)),
                double_dqn=bool(cfg.algo.get("double_dqn", True)),
            ),
            device=device,
        )

    if cfg.algo_name in ("a2c", "ppo"):
        hidden = tuple(cfg.network.get("hidden_sizes", (128, 128)))
        activation = cfg.network.get("activation", "tanh")
        device = resolve_device("auto")
        policy_net = GaussianPolicyNetwork(
            obs_dim, num_actions, hidden, activation,
            log_std_init=float(cfg.network.get("log_std_init", -0.5)),
        )
        # Критик — отдельная сеть, а не общее тело с актором: разделение
        # избавляет от подбора весов между двумя градиентами, идущими
        # в общие слои, и стоит на учебных задачах пренебрежимо дорого.
        value_net = MLPValueNetwork(obs_dim, tuple(cfg.network.get("value_hidden_sizes", hidden)),
                                    activation)
        shared = dict(
            gamma=float(cfg.algo["gamma"]),
            gae_lambda=float(cfg.algo["gae_lambda"]),
            learning_rate=float(learning_rate_schedule(cfg)(0)),
            value_coef=float(cfg.algo.get("value_coef", 0.5)),
            entropy_coef=float(cfg.algo.get("entropy_coef", 0.005)),
            max_grad_norm=float(cfg.algo.get("max_grad_norm", 0.5)),
            normalize_advantage=bool(cfg.algo.get("normalize_advantage", True)),
        )
        if cfg.algo_name == "a2c":
            return A2C(policy_net, value_net, A2CConfig(**shared), device=device)
        return PPO(
            policy_net, value_net,
            PPOConfig(
                clip_range=float(cfg.algo["clip_range"]),
                epochs=int(cfg.algo["epochs"]),
                minibatch_size=int(cfg.algo["minibatch_size"]),
                target_kl=float(cfg.algo.get("target_kl", 0.0)),
                **shared,
            ),
            device=device, seed=seed,
        )

    raise ValueError(
        f"алгоритм {cfg.algo_name!r} пока не поддержан скриптом; "
        "добавьте ветку в build_algo и цикл в labrl.train"
    )


def run_seed(cfg: ExperimentConfig, seed: int, quick: bool, worker_id: int = 0) -> dict:
    """Один прогон обучения на одном сиде. Возвращает сводку для отчёта.

    ``worker_id`` смещает порт связи с Unity. Разные значения позволяют
    держать несколько обучений одновременно; при одинаковых второй запуск
    не подключится к своей среде, а попытается занять чужой порт.
    """
    set_global_seed(seed)

    total_steps = cfg.total_steps
    if quick:
        total_steps = max(QUICK_MIN_STEPS, total_steps // QUICK_STEP_DIVISOR)

    run = create_run_dir(cfg.env_id, cfg.algo_name, seed)
    dump_config(cfg, run.config_yaml)
    run.write_pip_freeze()

    build_path = None if cfg.env["mode"] == "editor" else cfg.env["build_path"]
    handle = open_unity_env(
        cfg.env_id,
        build_path=build_path,
        seed=seed,
        time_scale=float(cfg.env.get("time_scale", 20.0)),
        no_graphics=bool(cfg.env.get("no_graphics", True)),
        env_parameters=cfg.env.get("env_parameters"),
        num_areas=int(cfg.env.get("num_areas", 1)),
        worker_id=worker_id,
    )

    try:
        vec = VecUnityEnv(handle)
        vec.reset()
        run.write_env_info({**handle.env_info(), "num_envs": vec.num_envs, "seed": seed,
                            "total_steps": total_steps, "quick": quick})

        discrete = tuple(vec.action_spec.discrete_branches)
        num_actions = int(discrete[0]) if discrete else int(vec.action_spec.continuous_size)
        algo = build_algo(cfg, vec.single_obs_dim, num_actions, seed)
        logger = TBLogger(run.tb)

        def report(step: int, reward: float, success: float) -> None:
            print(f"  шаг {step:>7}: Eval/Mean Reward {reward:+.4f}  Eval/Success Rate {success:.0%}",
                  flush=True)

        eval_every = int(cfg.eval["every_steps"])
        eval_episodes = int(cfg.eval["episodes"])
        # Порог успеха эпизода зависит от среды: «награда положительна» годится
        # для сред «дойди до цели», но не для сред «продержись как можно дольше»,
        # где любой эпизод даёт положительную сумму.
        success_threshold = float(cfg.eval.get("success_threshold", 0.0))
        schedule = build_schedule(cfg.algo["epsilon"])

        if cfg.algo_name in STRATEGIES:
            result = train_bandit(
                vec=vec, algo=algo, epsilon_schedule=schedule, logger=logger,
                cfg=BanditTrainConfig(total_steps=total_steps, eval_every_steps=eval_every,
                                      eval_episodes=eval_episodes,
                                      success_threshold=success_threshold),
                seed=seed, on_eval=report,
            )
        elif cfg.algo_name in ("a2c", "ppo"):
            result = train_on_policy(
                vec=vec, algo=algo, logger=logger,
                cfg=OnPolicyTrainConfig(
                    total_steps=total_steps,
                    rollout_steps=int(cfg.algo["rollout_steps"]),
                    eval_every_steps=eval_every,
                    eval_episodes=eval_episodes,
                    success_threshold=success_threshold,
                ),
                seed=seed, on_eval=report,
                lr_schedule=learning_rate_schedule(cfg),
            )
        elif cfg.algo_name == "qlearning":
            result = train_q_learning(
                vec=vec, algo=algo, epsilon_schedule=schedule, logger=logger,
                cfg=TabularTrainConfig(total_steps=total_steps, eval_every_steps=eval_every,
                                       eval_episodes=eval_episodes,
                                       success_threshold=success_threshold),
                seed=seed, on_eval=report,
                lr_schedule=learning_rate_schedule(cfg),
            )
        else:
            result = train_dqn(
                vec=vec, algo=algo, epsilon_schedule=schedule, logger=logger,
                cfg=DQNTrainConfig(
                    total_steps=total_steps,
                    buffer_size=int(cfg.algo["buffer_size"]),
                    learning_starts=int(cfg.algo["learning_starts"]),
                    train_freq=int(cfg.algo.get("train_freq", 1)),
                    gradient_steps=int(cfg.algo.get("gradient_steps", 1)),
                    eval_every_steps=eval_every,
                    eval_episodes=eval_episodes,
                    success_threshold=success_threshold,
                ),
                seed=seed, on_eval=report,
            )

        logger.hparams(
            {
                "seed": seed, "algo": cfg.algo_name, "env_id": cfg.env_id,
                "total_steps": total_steps, "num_envs": vec.num_envs,
                "git_commit": git_commit_hash(), **{k: v for k, v in cfg.algo.items() if not isinstance(v, dict)},
            },
            {"final/eval_reward": result.last_eval_reward,
             "final/eval_success": result.last_eval_success},
        )
        missing = logger.missing_required_tags()
        logger.dump_metrics(run.metrics_json, extra={
            "seed": seed,
            "episodes": len(result.episode_returns),
            "eval_history": result.eval_history,
            "wall_time_sec": result.wall_time,
            "missing_required_tags": list(missing),
        })
        logger.close()

        save_checkpoint(run.ckpt / "final.pt", algo.state_dict())

        onnx_path = _export(cfg, algo, run.onnx / "policy.onnx", vec)

        summary = {
            "seed": seed,
            "run_dir": str(run.root),
            "eval_reward": result.last_eval_reward,
            "eval_success": result.last_eval_success,
            "episodes": len(result.episode_returns),
            "wall_time_sec": round(result.wall_time, 1),
            "missing_required_tags": list(missing),
            "onnx": str(onnx_path) if onnx_path else None,
            "threshold": cfg.success_criteria.threshold,
            "passed": bool(result.last_eval_reward >= cfg.success_criteria.threshold),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        handle.close()


def _export(cfg: ExperimentConfig, algo, dest: Path, vec: VecUnityEnv) -> Path | None:
    """Экспортирует политику в ONNX и **блокирует** приёмку при провале проверки."""
    if not cfg.export.get("verify", True) and not cfg.export.get("onnx_path"):
        return None

    spec = ActionSpecLite(discrete_branches=tuple(vec.action_spec.discrete_branches),
                          continuous_size=vec.action_spec.continuous_size)
    obs_shapes = vec.obs_shapes
    policy = algo.policy_module()

    path = export_policy_to_onnx(policy, spec, obs_shapes, dest)

    if cfg.export.get("verify", True):
        # Наблюдения из реального распределения среды: требование 10.5 говорит
        # именно о нём, а не о случайном шуме.
        sample = _collect_sample_obs(vec, count=64)
        result = verify_onnx_model(path, policy, spec, obs_shapes, sample_obs=sample)
        print(result.report(), flush=True)
        result.raise_if_failed()

    final = resolve_path(cfg.export["onnx_path"])
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(path.read_bytes())
    print(f"модель скопирована в проект Unity: {final}", flush=True)
    return final


def _collect_sample_obs(vec: VecUnityEnv, count: int) -> list[np.ndarray]:
    """Собирает наблюдения случайной политикой — для проверки числового паритета.

    Требование 10.5 говорит именно о наблюдениях «из реального распределения
    среды», а не о случайном шуме: сеть может совпадать с ONNX на шуме
    и расходиться на данных, которые действительно встречаются.
    """
    rng = np.random.default_rng(0)
    branches = tuple(vec.action_spec.discrete_branches)
    continuous = int(vec.action_spec.continuous_size)
    collected: list[np.ndarray] = []
    obs = vec.reset()
    while sum(len(c) for c in collected) < count:
        collected.append(obs[0].copy())
        if branches:
            actions = np.stack(
                [rng.integers(0, b, size=vec.num_envs) for b in branches], axis=1
            ).astype(np.int32)
        else:
            actions = rng.uniform(-1.0, 1.0, size=(vec.num_envs, continuous)).astype(np.float32)
        obs = vec.step(actions).obs
    return [np.concatenate(collected, axis=0)[:count]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="путь к configs/E##_<Name>__<algo>.yaml")
    parser.add_argument("--seed", type=int, help="один сид; по умолчанию первый из конфига")
    parser.add_argument("--all-seeds", action="store_true", help="прогнать все сиды из конфига")
    parser.add_argument("--quick", action="store_true",
                        help=f"сокращённый бюджет (1/{QUICK_STEP_DIVISOR}, но не меньше {QUICK_MIN_STEPS} шагов); "
                             "смоук-тест конвейера — критерий приёмки при нём не проверяется")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="смещение порта связи с Unity; разные значения позволяют "
                             "запускать несколько обучений одновременно")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.seeds if args.all_seeds else [args.seed if args.seed is not None else cfg.seeds[0]]

    summaries = []
    for seed in seeds:
        print(f"\n=== {cfg.env_id} / {cfg.algo_name} / seed {seed}{' / quick' if args.quick else ''} ===", flush=True)
        summaries.append(run_seed(cfg, seed, args.quick, args.worker_id))

    passed = [s for s in summaries if s["passed"]]
    print(f"\nитог: критерий приёмки достигнут на {len(passed)} из {len(summaries)} сидов "
          f"(порог {cfg.success_criteria.threshold} по {cfg.success_criteria.metric})", flush=True)

    if args.quick:
        # Быстрый прогон — смоук-тест конвейера, а не проверка результата:
        # на сокращённом бюджете критерий приёмки не обязан достигаться,
        # и его недостижение ошибкой не является. Настоящие ошибки — исключение,
        # непройденная верификация ONNX, отсутствие обязательных тегов —
        # роняют прогон раньше, до этой строки.
        print("быстрый прогон: критерий приёмки не проверяется, "
              "проверялась работоспособность конвейера", flush=True)
        return 0

    return 0 if len(passed) == len(summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
