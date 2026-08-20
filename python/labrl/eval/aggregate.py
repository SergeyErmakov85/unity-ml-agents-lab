"""Агрегация результатов по сидам: IQM и доверительный интервал (требование 12.3).

Почему IQM, а не среднее. В RL распределение итоговых наград по сидам обычно
имеет тяжёлые хвосты: один удачный или один провальный сид сдвигает среднее
на величину, превышающую разницу между алгоритмами. Interquartile mean —
среднее по средним 50 % значений — устойчив к обоим хвостам и является
рекомендованной метрикой (Agarwal et al., 2021, «Deep RL at the Edge of the
Statistical Precipice»; тот же выбор делает библиотека ``rliable``).

Сравнение алгоритмов по одному прогону инструкцией запрещено (12.3), поэтому
все функции модуля принимают массив ``(num_seeds, ...)`` и не имеют варианта
для одного сида.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Число ресэмплов бутстрэпа по умолчанию.
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000

#: Уровень доверия по умолчанию.
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class IQMResult:
    """IQM с доверительным интервалом."""

    iqm: float
    ci_low: float
    ci_high: float
    confidence: float
    num_seeds: int
    num_samples: int

    def format(self) -> str:
        return f"IQM = {self.iqm:.4f}, {self.confidence:.0%} CI [{self.ci_low:.4f}, {self.ci_high:.4f}], сидов: {self.num_seeds}"


def interquartile_mean(values: np.ndarray) -> float:
    """Среднее по значениям между 25-м и 75-м процентилями.

    Реализация повторяет ``rliable``: значения сортируются, отбрасывается по
    четверти с каждого края (по числу элементов, а не по значению порога),
    усредняется остаток.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("пустой массив значений")
    ordered = np.sort(flat)
    lo = int(np.floor(flat.size / 4))
    hi = int(np.ceil(flat.size * 3 / 4))
    middle = ordered[lo:hi]
    if middle.size == 0:  # массив из 1–2 элементов
        middle = ordered
    return float(np.mean(middle))


def stratified_bootstrap_iqm(
    scores: np.ndarray,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> IQMResult:
    """IQM и доверительный интервал по **стратифицированному** бутстрэпу.

    Стратификация — по сидам: в каждом ресэмпле сначала выбираются сиды
    с возвращением, затем внутри каждого выбранного сида — его эпизоды
    с возвращением. Так интервал учитывает обе причины разброса: между сидами
    и внутри сида. Обычный бутстрэп по «всем эпизодам подряд» их смешивает
    и даёт слишком узкий интервал.

    Args:
        scores: ``(num_seeds, num_episodes)`` — награды за эпизод.
            Одномерный массив трактуется как ``(num_seeds, 1)``.
        resamples: число ресэмплов бутстрэпа.
        confidence: уровень доверия, например 0.95.
        seed: сид генератора ресэмплов (воспроизводимость интервала).
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"ожидался массив (num_seeds, num_episodes), получено {arr.shape}")
    num_seeds, num_episodes = arr.shape
    if num_seeds < 2:
        raise ValueError(
            f"нужно минимум 2 сида для доверительного интервала, получено {num_seeds}; "
            "инструкция требует минимум 3 (12.2)"
        )

    rng = np.random.default_rng(seed)
    point = interquartile_mean(arr)

    boot = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        seed_idx = rng.integers(0, num_seeds, size=num_seeds)
        ep_idx = rng.integers(0, num_episodes, size=(num_seeds, num_episodes))
        sample = arr[seed_idx[:, None], ep_idx]
        boot[i] = interquartile_mean(sample)

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return IQMResult(
        iqm=point,
        ci_low=float(lo),
        ci_high=float(hi),
        confidence=confidence,
        num_seeds=num_seeds,
        num_samples=int(arr.size),
    )
