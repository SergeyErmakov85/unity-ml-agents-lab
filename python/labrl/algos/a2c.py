"""A2C — Advantage Actor-Critic. Реализация с нуля на PyTorch.

Идея. Value-based методы (`E03`) выбирают действие как ``argmax_a Q(s, a)``;
в непрерывном пространстве такого перебора нет. Policy gradient идёт другим
путём: политика **параметризуется напрямую**, а её параметры сдвигаются
в сторону роста ожидаемой награды::

    ∇_θ J = E[ ∇_θ log π_θ(a|s) · Ψ_t ]

Весь вопрос в том, что взять за ``Ψ_t``. Сырая сумма наград эпизода
(REINFORCE) даёт несмещённую, но чудовищно шумную оценку: действие
вознаграждается за всё, что случилось после него, включая не связанное с ним.
Actor-Critic заменяет её на **преимущество** ``A(s, a) = Q(s, a) − V(s)`` —
«насколько это действие лучше среднего в этом состоянии». Вычесть ``V(s)``
можно без смещения (это baseline, не зависящий от действия), а дисперсия
падает на порядок.

Отсюда две сети:

* **актор** ``π_θ(a|s)`` — гауссова политика, выдаёт среднее действия;
* **критик** ``V_φ(s)`` — оценивает ценность состояния, нужен только
  для расчёта преимущества.

Функция потерь::

    L = −E[ log π(a|s) · Â ]  +  c_v · E[ (V(s) − R)² ]  −  c_H · H(π)

Три слагаемых: подъём вероятности действий с положительным преимуществом,
обучение критика на цель ``R = Â + V(s)`` и **бонус за энтропию**, который
не даёт политике схлопнуться в детерминированную раньше, чем она что-то
выучит.

Чем A2C отличается от PPO. Здесь на собранных данных делается **ровно один**
шаг градиента, после чего они выбрасываются: формула градиента верна только
для той политики, которой данные собраны. PPO (`labrl.algos.ppo`) снимает это
ограничение, добавляя отношение правдоподобий и обрезку, и потому может
пройти по одному роллауту несколько раз.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labrl.buffers.rollout import RolloutBatch
from labrl.nets.gaussian_policy import to_env_action


@dataclass
class A2CConfig:
    """Все гиперпараметры метода. «Магических чисел» внутри методов нет (8.6)."""

    #: Коэффициент дисконтирования, безразмерный, [0, 1).
    gamma: float = 0.99
    #: Параметр λ обобщённого преимущества (GAE), [0, 1].
    gae_lambda: float = 0.95
    #: Скорость обучения общего оптимизатора Adam.
    learning_rate: float = 3e-4
    #: Вес ошибки критика в общей функции потерь.
    value_coef: float = 0.5
    #: Вес бонуса за энтропию. Ноль означает «политике разрешено схлопнуться».
    entropy_coef: float = 0.005
    #: Максимальная норма градиента; 0 — не клипировать.
    max_grad_norm: float = 0.5
    #: Нормировать ли преимущества по батчу.
    #:
    #: По умолчанию **выключено** — так же, как в эталонных реализациях A2C.
    #: Нормировка убирает зависимость масштаба градиента от масштаба награды,
    #: но батч A2C мал (одно обновление на короткий роллаут), поэтому среднее
    #: и разброс по нему сами оценены грубо. Деление на такой разброс
    #: превращает шум критика в полноразмерный градиент — и после того, как
    #: задача решена и преимущества выродились, политика начинает случайно
    #: блуждать. Измерено на `E04_BallBalance`: с нормировкой прогон закончился
    #: на 45.3 из 100, без неё — см. docs/envs/E04_BallBalance.md.
    #: У PPO батч на порядок больше, а обрезка ограничивает шаг, поэтому там
    #: нормировка включена.
    normalize_advantage: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda должна быть в [0, 1], получено {self.gae_lambda}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate должен быть > 0, получено {self.learning_rate}")


class ActOutput(NamedTuple):
    """Что возвращает :meth:`A2C.act`.

    Attributes:
        env_action: действие в диапазоне Unity ``[-1, 1]`` — его получает среда.
        raw_action: сэмпл политики **до** приведения к диапазону; именно он
            хранится в роллауте, потому что log π считается для него.
        log_prob: log π(a|s) на момент сбора.
        value: V(s) на момент сбора.
    """

    env_action: np.ndarray
    raw_action: np.ndarray
    log_prob: np.ndarray
    value: np.ndarray


class A2C:
    """Advantage Actor-Critic для непрерывных действий.

    Args:
        policy_net: актор. Любой ``nn.Module`` с методами гауссовой политики
            (``forward -> среднее``, ``log_prob``, ``entropy``) — см.
            :class:`labrl.nets.gaussian_policy.GaussianPolicyNetwork`.
            Архитектуру задаёт пользователь (требование 8.4).
        value_net: критик ``(B, obs_dim) -> (B, 1)``.
        cfg: гиперпараметры.
        device: устройство вычислений.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        cfg: A2CConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or A2CConfig()
        self.device = torch.device(device)

        self.policy_net = policy_net.to(self.device)
        self.value_net = value_net.to(self.device)

        # Один оптимизатор на обе сети: функция потерь общая, и разделять шаги
        # незачем — это лишь усложнило бы согласование их темпов.
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.cfg.learning_rate,
        )
        self.updates = 0

    # --- взаимодействие со средой ---------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> ActOutput:
        """Сэмплирует действие из политики для батча наблюдений.

        Шум берётся из переданного генератора numpy, а не из глобального
        состояния torch: воспроизводимость прогона не должна зависеть от того,
        сколько раз кто-то ещё дёрнул torch.randn.
        """
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        mean = self.policy_net(tensor)
        std = self.policy_net.clamped_log_std().exp()

        noise = torch.as_tensor(
            rng.standard_normal(size=tuple(mean.shape)).astype(np.float32), device=self.device
        )
        raw = mean + std * noise

        log_prob = self.policy_net.log_prob(tensor, raw)
        value = self.value_net(tensor).squeeze(-1)

        raw_np = raw.cpu().numpy()
        return ActOutput(
            env_action=to_env_action(raw_np),
            raw_action=raw_np,
            log_prob=log_prob.cpu().numpy(),
            value=value.cpu().numpy(),
        )

    @torch.no_grad()
    def deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Действие детерминированной политики — среднее, приведённое к [-1, 1].

        Это **в точности** то, что вычисляет экспортированный граф ONNX
        (`deterministic_continuous_actions`), поэтому оценка в Python и
        поведение в Unity сравнимы напрямую (требование 10.6).
        """
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        mean = self.policy_net(tensor).cpu().numpy()
        return to_env_action(mean)

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        """``V(s)`` для батча наблюдений — нужен для бутстрэппинга в GAE."""
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.value_net(tensor).squeeze(-1).cpu().numpy()

    # --- обучение --------------------------------------------------------

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """Единственное место, где меняются параметры (требование 8.7).

        Один шаг градиента по всему роллауту. Повторить его на тех же данных
        нельзя: после шага политика уже другая, и формула policy gradient
        для этих данных перестаёт быть верной. Именно это ограничение снимает
        PPO.
        """
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=self.device)
        advantage = torch.as_tensor(batch.advantage, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        old_log_prob = torch.as_tensor(batch.log_prob, dtype=torch.float32, device=self.device)

        if self.cfg.normalize_advantage and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        log_prob = self.policy_net.log_prob(obs, action)
        entropy = self.policy_net.entropy()
        value = self.value_net(obs).squeeze(-1)

        # Знак минус: оптимизатор минимизирует, а policy gradient — подъём.
        policy_loss = -(log_prob * advantage).mean()
        value_loss = F.mse_loss(value, returns)
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf"),
        )
        self.optimizer.step()
        self.updates += 1

        with torch.no_grad():
            # Приближённая KL между политикой до и после шага — индикатор того,
            # что шаг не был разрушительным.
            approx_kl = (old_log_prob - log_prob).mean()
            explained = explained_variance(value.cpu().numpy(), batch.returns)

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl.item()),
            "explained_variance": float(explained),
            "grad_norm": float(grad_norm),
            "log_std": float(self.policy_net.clamped_log_std().mean().item()),
            "updates": float(self.updates),
        }

    def set_learning_rate(self, learning_rate: float) -> None:
        """Меняет шаг Adam — для расписания скорости обучения.

        Зачем он вообще нужен on-policy методу. Когда задача решена, награда
        среды становится почти постоянной, все преимущества обращаются в ноль,
        и единственное, что остаётся в оценке преимущества, — ошибка критика.
        Нормировка преимуществ по батчу возводит этот шум обратно в единичный
        масштаб, и политика начинает случайно блуждать — вплоть до полного
        разрушения уже выученного поведения. Затухающий шаг делает блуждание
        всё более мелким и оставляет политику там, где она сошлась
        (docs/07_TROUBLESHOOTING.md, T-13).

        Так же поступает и штатный тренер ML-Agents: `learning_rate_schedule:
        linear` — значение по умолчанию его конфигов.
        """
        if learning_rate < 0.0:
            raise ValueError(f"learning_rate должен быть >= 0, получено {learning_rate}")
        self.cfg.learning_rate = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    # --- сохранение и экспорт -------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(state["policy_net"])
        self.value_net.load_state_dict(state["value_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))

    def policy_module(self) -> nn.Module:
        """Модуль, экспортируемый в ONNX.

        Это актор: обёртка экспорта берёт от него среднее и приводит его
        к диапазону Unity. Критик в Unity не нужен — он существовал только
        ради расчёта преимущества при обучении.
        """
        return self.policy_net.eval()


def explained_variance(predicted: np.ndarray, target: np.ndarray) -> float:
    """Доля дисперсии цели, объяснённая критиком: ``1 − Var(y − ŷ)/Var(y)``.

    Главная диагностика критика. Единица — идеальное предсказание, ноль —
    «не лучше среднего», отрицательное значение — «хуже, чем константа»,
    и почти всегда означает слишком большой шаг обучения.
    """
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    var = target.var()
    if var <= 0.0:
        return 0.0
    return float(1.0 - (target - predicted).var() / var)
