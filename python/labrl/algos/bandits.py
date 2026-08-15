"""Многорукие бандиты: ε-greedy, UCB1, Thompson Sampling. Реализация с нуля.

Урок 1.7. Бандит — MDP с одним состоянием и эпизодом длины 1: нет ни
следующего состояния, ни дисконтирования, ни бутстрэппинга. Остаётся ровно
одна задача — **как делить бюджет попыток** между проверкой неизвестного
(exploration) и использованием лучшего известного (exploitation).

Общая часть у всех трёх стратегий одна: оценка средней награды руки как
выборочное среднее::

    Q_k = S_k / N_k          S_k — сумма наград руки k, N_k — число нажатий

Обновление ведётся инкрементально, ``Q ← Q + (r − Q)/N``: хранить всю историю
не нужно, а шаг ``1/N`` — это в точности несмещённое выборочное среднее.

Различаются стратегии **только правилом выбора**.

**ε-greedy.** С вероятностью ε — случайная рука, иначе argmax Q. Разведка
равномерна и не затухает сама: ε задаётся расписанием снаружи. Простейший
метод и худший из трёх по сожалению: он одинаково часто перепроверяет
и почти-лучшую руку, и заведомо провальную.

**UCB1** (Auer et al., 2002). Выбирается рука с максимальной *оптимистичной*
оценкой::

    UCB_k = Q_k + c · sqrt( ln t / N_k )

Второе слагаемое — ширина доверительного интервала: оно велико для рук,
испробованных редко, и стягивается по мере накопления данных. «Оптимизм
в условиях неопределённости»: разведка направлена туда, где оценка неточна,
а не куда попало. Руки, не нажатые ни разу, получают бесконечный приоритет.

**Thompson Sampling** (Thompson, 1933). Байесовский подход: для каждой руки
поддерживается апостериорное распределение её вероятности награды, из него
берётся **один сэмпл**, и выбирается рука с наибольшим сэмплом. Для
бернуллиевских наград сопряжённое априорное — Beta, и апостериорное считается
точно::

    θ_k ~ Beta(α₀ + успехи_k, β₀ + промахи_k)

Чем меньше данных о руке, тем шире её распределение и тем чаще она случайно
выигрывает розыгрыш — разведка возникает сама, без отдельного параметра.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.nn as nn

#: Поддерживаемые стратегии выбора руки.
STRATEGIES = ("eps_greedy", "ucb", "thompson")


@dataclass
class BanditConfig:
    """Все гиперпараметры метода. «Магических чисел» внутри методов нет (8.6)."""

    #: Правило выбора руки: ``eps_greedy`` | ``ucb`` | ``thompson``.
    strategy: str = "eps_greedy"
    #: Коэффициент оптимизма UCB1. Классическое значение c = √2 ≈ 1.414;
    #: больше — шире разведка, дольше сходимость.
    ucb_c: float = 1.414
    #: Параметры априорного Beta(α₀, β₀) для Thompson. (1, 1) — равномерное
    #: распределение на [0, 1], то есть «о руке ничего не известно».
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    #: Начальная оценка ненажатой руки. Положительное значение даёт
    #: «оптимизм в условиях неопределённости» и само по себе стимулирует
    #: разведку даже при ε = 0.
    initial_value: float = 0.0

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy должна быть одной из {STRATEGIES}, получено {self.strategy!r}")
        if self.ucb_c < 0.0:
            raise ValueError(f"ucb_c должен быть >= 0, получено {self.ucb_c}")
        if self.prior_alpha <= 0.0 or self.prior_beta <= 0.0:
            raise ValueError("параметры Beta-априорного должны быть > 0")


@dataclass
class BanditBatch:
    """Батч исходов: какие руки нажаты и что они вернули.

    Батч, а не одиночное нажатие, потому что среда векторизована: K арен дают
    до K исходов за шаг.
    """

    action: np.ndarray  # (B,) индексы рук
    reward: np.ndarray  # (B,) награды, для бернуллиевской среды — 0.0 или 1.0


class Bandit:
    """Оценка ценности рук плюс одно из трёх правил выбора.

    Args:
        num_actions: число рук.
        cfg: гиперпараметры.

    Состояние алгоритма — три вектора длины `num_actions`: число нажатий,
    сумма наград и (для Thompson) число успехов. Нейросети здесь нет: она
    появляется только на этапе экспорта, где оценки оборачиваются линейным
    слоем (см. :mod:`labrl.nets.tabular`).
    """

    def __init__(self, num_actions: int, cfg: BanditConfig | None = None) -> None:
        self.cfg = cfg or BanditConfig()
        self.num_actions = int(num_actions)
        if self.num_actions < 2:
            raise ValueError(f"рук должно быть >= 2, получено {self.num_actions}")

        self.counts = np.zeros(self.num_actions, dtype=np.int64)
        self.values = np.full(self.num_actions, self.cfg.initial_value, dtype=np.float64)
        #: Число наград, равных 1 — достаточная статистика Beta-апостериорного.
        self.successes = np.zeros(self.num_actions, dtype=np.int64)
        self.pulls = 0
        self.updates = 0

    # --- взаимодействие со средой ---------------------------------------

    def act(self, batch_size: int, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        """Выбирает руку для каждого из `batch_size` слотов.

        Args:
            batch_size: сколько арен ждут действия.
            epsilon: вероятность случайной руки; используется только
                стратегией ``eps_greedy``, остальные её игнорируют.
            rng: генератор — передаётся явно ради воспроизводимости.

        Returns:
            ``(batch_size,)`` индексы рук.
        """
        if self.cfg.strategy == "eps_greedy":
            greedy = np.full(batch_size, int(np.argmax(self.values)), dtype=np.int64)
            if epsilon <= 0.0:
                return greedy
            explore = rng.random(batch_size) < epsilon
            random_arms = rng.integers(0, self.num_actions, size=batch_size)
            return np.where(explore, random_arms, greedy)

        if self.cfg.strategy == "ucb":
            # Правило детерминировано, поэтому все слоты шага выбирают одну
            # и ту же руку: статистика обновится только после шага. Это и есть
            # батчевый UCB — K нажатий рекомендованной руки за раз.
            return np.full(batch_size, self.ucb_scores().argmax(), dtype=np.int64)

        # thompson: каждому слоту — свой розыгрыш апостериорного.
        alpha = self.cfg.prior_alpha + self.successes
        beta = self.cfg.prior_beta + (self.counts - self.successes)
        samples = rng.beta(alpha, beta, size=(batch_size, self.num_actions))
        return samples.argmax(axis=1).astype(np.int64)

    def greedy_action(self, batch_size: int) -> np.ndarray:
        """Жадный выбор ``argmax_k Q_k`` — итоговая политика, она же уходит в ONNX.

        Разведка здесь отсутствует у **всех** трёх стратегий: разведка нужна,
        пока идёт обучение, а приёмочное число измеряется по политике,
        которая будет работать в Unity.
        """
        return np.full(batch_size, int(np.argmax(self.values)), dtype=np.int64)

    def ucb_scores(self) -> np.ndarray:
        """Оптимистичные оценки UCB1. Ненажатая рука получает +∞."""
        scores = np.full(self.num_actions, np.inf)
        tried = self.counts > 0
        if not tried.any():
            return scores
        # ln(t) при t = 0 не определён; счётчик начинается с 1.
        log_t = math.log(max(self.pulls, 1))
        scores[tried] = self.values[tried] + self.cfg.ucb_c * np.sqrt(log_t / self.counts[tried])
        return scores

    # --- обучение --------------------------------------------------------

    def update(self, batch: BanditBatch) -> dict[str, float]:
        """Единственное место, где меняются оценки (требование 8.7).

        Обновление инкрементальное и **последовательное внутри батча**: если
        две арены нажали одну руку, второй исход должен уточнять уже уточнённую
        оценку, а не исходную. Векторизовать это нельзя без потери точного
        совпадения с однопоточным прогоном, а бюджет батча — единицы элементов.

        Returns:
            Метрики шага обучения для TensorBoard.
        """
        action = np.asarray(batch.action, dtype=np.int64)
        reward = np.asarray(batch.reward, dtype=np.float64)

        if action.size == 0:
            return {
                "value_error_abs": 0.0,
                "q_max": float(self.values.max()),
                "q_mean": float(self.values.mean()),
                "mean_step_size": 0.0,
                "updates": float(self.updates),
            }

        errors = np.empty(action.size)
        step_sizes = np.empty(action.size)

        for i, (arm, r) in enumerate(zip(action, reward)):
            self.counts[arm] += 1
            self.pulls += 1
            if r > 0.0:
                self.successes[arm] += 1

            step = 1.0 / self.counts[arm]
            error = r - self.values[arm]
            self.values[arm] += step * error

            errors[i] = abs(error)
            step_sizes[i] = step

        self.updates += action.size

        return {
            "value_error_abs": float(errors.mean()),
            "q_max": float(self.values.max()),
            "q_mean": float(self.values.mean()),
            "mean_step_size": float(step_sizes.mean()),
            "updates": float(self.updates),
        }

    # --- сохранение и экспорт -------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.copy(),
            "counts": self.counts.copy(),
            "successes": self.successes.copy(),
            "pulls": self.pulls,
            "updates": self.updates,
            "cfg": self.cfg,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        values = np.asarray(state["values"], dtype=np.float64)
        if values.shape != (self.num_actions,):
            raise ValueError(
                f"форма оценок {values.shape} не совпадает с ожидаемой {(self.num_actions,)}"
            )
        self.values = values.copy()
        self.counts = np.asarray(state["counts"], dtype=np.int64).copy()
        self.successes = np.asarray(state["successes"], dtype=np.int64).copy()
        self.pulls = int(state.get("pulls", 0))
        self.updates = int(state.get("updates", 0))

    def policy_module(self) -> nn.Module:
        """Модуль, экспортируемый в ONNX.

        Бандит — MDP с одним состоянием, и наблюдение среды `E00_Bandit` равно
        константе 1.0, то есть корректному one-hot вектору длины 1. Значит
        таблица оценок формы (1, K) экспортируется тем же линейным слоем без
        смещения, что и в `E01_GridWorld`: ``onehot(s) @ Qᵀ == Q[s]``.
        В Unity уезжает **та же самая** политика, а не её приближение.
        """
        from labrl.nets.tabular import OneHotQTable

        return OneHotQTable.from_table(self.values[None, :])

    def regret_per_pull(self, arm_probabilities: np.ndarray) -> float:
        """Среднее сожаление за нажатие при известных вероятностях рук.

        Диагностическая величина: в реальной задаче вероятности неизвестны,
        но в учебной среде они заданы ТЗ, и сожаление — честная мера качества
        разведки, в отличие от шумной средней награды.
        """
        p = np.asarray(arm_probabilities, dtype=np.float64)
        if self.pulls == 0:
            return 0.0
        return float((p.max() - p) @ self.counts / self.pulls)
