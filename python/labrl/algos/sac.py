"""SAC — Soft Actor-Critic (Haarnoja et al., 2018). Реализация с нуля.

Идея. Обычный RL максимизирует ожидаемую сумму наград. SAC максимизирует её
**вместе с энтропией политики**::

    J(π) = Σ_t E[ r(s_t, a_t) + α·H(π(·|s_t)) ]

Смысл добавки: агенту выгодно быть настолько случайным, насколько это
не мешает набирать награду. Из этого следует всё остальное — разведка
встроена в цель, а не приделана снаружи ε-жадностью; политика не схлопывается
в первый найденный локальный оптимум; обучение устойчиво к неудачной
инициализации.

Три механизма, без которых метод не работает
--------------------------------------------

**1. Два критика и минимум из них.** Бутстрэп-обучение Q систематически
**завышает** ценность: в цели стоит максимум по зашумлённым оценкам, а
максимум шума смещён вверх. SAC держит два независимых критика и берёт
в цели ``min(Q₁, Q₂)`` — намеренно пессимистичную оценку (приём из TD3).

**2. Reparameterization trick.** Действие берётся как ``a = tanh(μ + σ·ε)``,
где ε — независимый шум. Благодаря этому действие является гладкой функцией
параметров, и градиент Q по действию течёт обратно в веса политики. Оценка
градиента получается на порядок менее шумной, чем у REINFORCE.

**3. Автоподстройка α.** Температура α — «ручка» между наградой и энтропией,
и вручную её подбирать бессмысленно: нужное значение зависит от масштаба
награды, который меняется по ходу обучения. Вместо этого α настраивается
на **целевую энтропию** (по умолчанию −dim(A))::

    L(α) = −log α · ( log π(a|s) + H_target )

Если политика слишком детерминирована (log π велик), α растёт и возвращает
разведку; если слишком случайна — падает.

Правило обновления критика (soft Bellman)::

    y = r + γ·(1 − terminated)·[ min(Q̄₁, Q̄₂)(s′, a′) − α·log π(a′|s′) ],
        a′ ~ π(·|s′)
    L(Q_i) = MSE( Q_i(s, a), y )

Правило обновления актора::

    L(π) = E[ α·log π(a|s) − min(Q₁, Q₂)(s, a) ],  a ~ π(·|s) через reparam

Целевые сети критиков обновляются мягко (polyak): ``θ̄ ← τθ + (1−τ)θ̄``.
Жёсткое копирование, как в DQN, здесь работает хуже: цель меняется скачком,
а актор обучается на ней непрерывно.

Про ``(1 − terminated)``: будущее обнуляется только при истинном завершении.
Обрыв по ``MaxStep`` — не терминальное состояние, там бутстрэппинг обязателен.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labrl.buffers.replay import Batch
from labrl.nets.squashed_gaussian import SquashedGaussianPolicy


@dataclass
class SACConfig:
    """Все гиперпараметры метода. «Магических чисел» внутри методов нет (8.6)."""

    #: Коэффициент дисконтирования, безразмерный, [0, 1).
    gamma: float = 0.99
    #: Скорость обучения Adam — одна на актора, критиков и температуру.
    learning_rate: float = 3e-4
    #: Размер батча, переходов.
    batch_size: int = 256
    #: Коэффициент мягкого обновления целевых критиков: θ̄ ← τθ + (1−τ)θ̄.
    tau: float = 0.005
    #: Начальное значение температуры α.
    init_alpha: float = 0.2
    #: Подстраивать ли α автоматически. Ручной подбор возможен, но нужное
    #: значение зависит от масштаба награды и меняется по ходу обучения.
    autotune_alpha: bool = True
    #: Целевая энтропия. ``None`` — взять −dim(A), значение из статьи.
    target_entropy: float | None = None
    #: Максимальная норма градиента; 0 — не клипировать.
    max_grad_norm: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate должен быть > 0, получено {self.learning_rate}")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau должен быть в (0, 1], получено {self.tau}")
        if self.init_alpha <= 0.0:
            raise ValueError(f"init_alpha должен быть > 0, получено {self.init_alpha}")


class SAC:
    """Soft Actor-Critic для непрерывных действий.

    Args:
        policy_net: актор — squashed-гауссова политика
            (:class:`labrl.nets.squashed_gaussian.SquashedGaussianPolicy`).
        q1_net, q2_net: два критика ``(obs, action) -> (B,)``. Архитектуру
            задаёт пользователь (требование 8.4); они обязаны быть
            **независимо инициализированы** — в этом весь смысл минимума
            из двух оценок.
        cfg: гиперпараметры.
        device: устройство вычислений.

    Целевые критики создаются глубоким копированием переданных: просить
    у пользователя четыре сети значило бы просить его же продублировать
    архитектуру, а любая опечатка в дубле проявилась бы как «SAC почему-то
    не сходится».
    """

    def __init__(
        self,
        policy_net: SquashedGaussianPolicy,
        q1_net: nn.Module,
        q2_net: nn.Module,
        cfg: SACConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or SACConfig()
        self.device = torch.device(device)

        self.policy_net = policy_net.to(self.device)
        self.q1 = q1_net.to(self.device)
        self.q2 = q2_net.to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for param in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            param.requires_grad_(False)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.cfg.learning_rate)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.cfg.learning_rate)

        self.action_dim = int(policy_net.action_dim)
        self.target_entropy = (
            float(self.cfg.target_entropy) if self.cfg.target_entropy is not None
            else -float(self.action_dim)
        )
        # Оптимизируется логарифм α: сама α обязана быть положительной,
        # а на её логарифм ограничений нет.
        self.log_alpha = torch.tensor(
            [float(np.log(self.cfg.init_alpha))], device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.cfg.learning_rate)

        self.updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    # --- взаимодействие со средой ---------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Сэмплирует действие из политики. ``(B, obs_dim) -> (B, action_dim)``.

        Шум берётся из переданного генератора numpy, а не из глобального
        состояния torch: воспроизводимость прогона не должна зависеть от того,
        сколько раз кто-то ещё дёрнул torch.randn.
        """
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        noise = torch.as_tensor(
            rng.standard_normal(size=(tensor.shape[0], self.action_dim)).astype(np.float32),
            device=self.device,
        )
        action, _ = self.policy_net.sample(tensor, noise)
        return action.cpu().numpy()

    @torch.no_grad()
    def deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """``tanh(μ)`` — то же действие, которое выдаёт экспортированный граф."""
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.policy_net(tensor).cpu().numpy()

    def set_learning_rate(self, learning_rate: float) -> None:
        """Меняет шаг всех трёх оптимизаторов — для расписания."""
        if learning_rate < 0.0:
            raise ValueError(f"learning_rate должен быть >= 0, получено {learning_rate}")
        self.cfg.learning_rate = float(learning_rate)
        for optimizer in (self.policy_optimizer, self.q_optimizer, self.alpha_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate)

    # --- обучение --------------------------------------------------------

    def update(self, batch: Batch) -> dict[str, float]:
        """Единственное место, где меняются параметры (требование 8.7)."""
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.bool, device=self.device)

        alpha = self.alpha

        # --- критики ------------------------------------------------------
        with torch.no_grad():
            next_action, next_log_prob = self.policy_net.sample(next_obs)
            # Минимум из двух целевых критиков гасит систематическую переоценку.
            next_q = torch.min(
                self.q1_target(next_obs, next_action),
                self.q2_target(next_obs, next_action),
            )
            # Энтропийная добавка входит в цель: агент ценит не только награду,
            # но и свободу выбора в следующем состоянии.
            soft_next_value = next_q - alpha * next_log_prob
            target = reward + self.cfg.gamma * soft_next_value * (~terminated).float()

        q1_pred = self.q1(obs, action)
        q2_pred = self.q2(obs, action)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad_norm = self._clip(list(self.q1.parameters()) + list(self.q2.parameters()))
        self.q_optimizer.step()

        # --- актор --------------------------------------------------------
        # Критики на этом шаге не обучаются: градиент политики течёт через них,
        # но их веса меняет только q_optimizer.
        sampled_action, log_prob = self.policy_net.sample(obs)
        q_min = torch.min(self.q1(obs, sampled_action), self.q2(obs, sampled_action))
        policy_loss = (alpha * log_prob - q_min).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        policy_grad_norm = self._clip(list(self.policy_net.parameters()))
        self.policy_optimizer.step()

        # --- температура --------------------------------------------------
        alpha_loss = torch.zeros((), device=self.device)
        if self.cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # --- мягкое обновление целевых сетей ------------------------------
        self._polyak(self.q1, self.q1_target)
        self._polyak(self.q2, self.q2_target)
        self.updates += 1

        return {
            "loss": float(q_loss.item() + policy_loss.item()),
            "value_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
            "entropy": float(-log_prob.mean().item()),
            "q_mean": float(q1_pred.mean().item()),
            "q_max": float(q1_pred.max().item()),
            "td_error_abs": float((target - q1_pred).abs().mean().item()),
            "grad_norm": float(q_grad_norm),
            "policy_grad_norm": float(policy_grad_norm),
            "updates": float(self.updates),
        }

    def _clip(self, params: list[torch.nn.Parameter]) -> float:
        limit = self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf")
        return float(torch.nn.utils.clip_grad_norm_(params, limit))

    def _polyak(self, source: nn.Module, target: nn.Module) -> None:
        """θ̄ ← τθ + (1−τ)θ̄ — мягкое обновление целевой сети."""
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.mul_(1.0 - self.cfg.tau).add_(param, alpha=self.cfg.tau)

    # --- сохранение и экспорт -------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(state["policy_net"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.updates = int(state.get("updates", 0))

    def policy_module(self) -> nn.Module:
        """Модуль, экспортируемый в ONNX.

        Это актор, домноженный на 3: обёртка контракта делит выход на 3, и
        в графе окажется ровно ``tanh(μ)`` — то же действие, которое выдаёт
        :meth:`deterministic_action`. Критики в Unity не нужны.
        """
        return self.policy_net.export_module()
