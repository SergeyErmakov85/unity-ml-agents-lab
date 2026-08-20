"""Оценка политики: прогон заданного числа эпизодов и сбор метрик.

Оценка отделена от обучения намеренно. При обучении политика стохастична
(ε-жадность, сэмплирование из распределения), и её награда систематически
ниже настоящей. Приёмочные числа примера (`Eval/Mean Reward`, `Eval/Success Rate`,
сравнение с Unity по 10.6) считаются **детерминированной** политикой — той же,
что уйдёт в ONNX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from labrl.envs.vec_unity_env import VecUnityEnv

#: Политика: список наблюдений по сенсорам -> действия ``(N, …)``.
#: Строки неактивных слотов игнорируются средой, но массив обязан быть полным.
PolicyFn = Callable[[Sequence[np.ndarray]], np.ndarray]


@dataclass
class EvalResult:
    """Итог оценки."""

    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    success_rate: float
    truncated_fraction: float

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns))

    @property
    def std_return(self) -> float:
        return float(np.std(self.episode_returns))

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths))

    def format(self) -> str:
        return (
            f"эпизодов: {len(self.episode_returns)}, "
            f"награда {self.mean_return:.4f} ± {self.std_return:.4f}, "
            f"длина {self.mean_length:.1f}, "
            f"успех {self.success_rate:.1%}, "
            f"оборвано по MaxStep {self.truncated_fraction:.1%}"
        )


def evaluate(
    vec: VecUnityEnv,
    policy: PolicyFn,
    episodes: int = 20,
    max_steps: int | None = None,
    success_predicate: Callable[[float], bool] | None = None,
) -> EvalResult:
    """Прогоняет политику до накопления ``episodes`` завершённых эпизодов.

    Args:
        vec: векторизованная среда.
        policy: детерминированная политика.
        episodes: сколько эпизодов собрать. Требование 10.6 — 20 для сравнения
            Python и Unity; требование 12.2 — оценка на 3 сидах.
        max_steps: предохранитель от зависания, если среда перестала завершать
            эпизоды. ``None`` — ограничение ``200 × episodes`` шагов.
        success_predicate: что считать успехом по итоговой награде эпизода.
            ``None`` — успехом считается положительная суммарная награда.

    Returns:
        :class:`EvalResult`. Собираются **только завершённые** эпизоды:
        незавершённые в статистику не попадают, иначе среднее окажется
        смещённым вниз обрезанными траекториями.
    """
    if episodes <= 0:
        raise ValueError(f"episodes должно быть > 0, получено {episodes}")

    is_success = success_predicate or (lambda total: total > 0.0)
    budget = max_steps if max_steps is not None else 200 * episodes

    # reset() здесь безопасен: если среда уже сброшена, обёртка пропустит
    # вызов, а лишний env.reset() стоил бы одного шага с нулевым действием
    # (см. VecUnityEnv.reset и T-7 в docs/07_TROUBLESHOOTING.md).
    obs = vec.reset()
    n = vec.num_envs

    running_return = np.zeros(n, dtype=np.float64)
    running_length = np.zeros(n, dtype=np.int64)

    returns: list[float] = []
    lengths: list[int] = []
    truncations: list[bool] = []

    for _ in range(budget):
        if len(returns) >= episodes:
            break

        actions = policy(obs)
        result = vec.step(actions)

        running_return += result.reward
        running_length += result.active | result.done

        for slot in np.flatnonzero(result.done):
            returns.append(float(running_return[slot]))
            lengths.append(int(running_length[slot]))
            truncations.append(bool(result.truncated[slot]))
            running_return[slot] = 0.0
            running_length[slot] = 0

        obs = result.obs

    if not returns:
        raise RuntimeError(
            f"за {budget} шагов не завершился ни один эпизод; "
            "проверьте MaxStep агента и условия завершения в среде"
        )

    returns_arr = np.array(returns[:episodes], dtype=np.float64)
    lengths_arr = np.array(lengths[:episodes], dtype=np.int64)
    truncated_arr = np.array(truncations[:episodes], dtype=bool)

    return EvalResult(
        episode_returns=returns_arr,
        episode_lengths=lengths_arr,
        success_rate=float(np.mean([is_success(r) for r in returns_arr])),
        truncated_fraction=float(np.mean(truncated_arr)),
    )
