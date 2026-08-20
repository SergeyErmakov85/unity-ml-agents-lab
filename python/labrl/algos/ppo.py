"""PPO — Proximal Policy Optimization (Schulman et al., 2017). Реализация с нуля.

Проблема, которую решает PPO. У A2C (`labrl.algos.a2c`) собранный роллаут
можно использовать **ровно один раз**: формула policy gradient верна только
для той политики, которой данные собраны. Это расточительно — взаимодействие
со средой обычно дороже градиентного шага. Наивное лекарство «сделаем десять
шагов вместо одного» разрушает обучение: после первого шага политика уже
другая, и оставшиеся девять двигают её по неверной оценке, часто далеко
и необратимо.

Решение — importance sampling с **обрезкой**. Отношение правдоподобий::

    r_t(θ) = π_θ(a_t|s_t) / π_старая(a_t|s_t)

позволяет честно пересчитать градиент под новую политику, а обрезка не даёт
уйти далеко от старой::

    L^CLIP = E[ min( r_t·Â_t , clip(r_t, 1−ε, 1+ε)·Â_t ) ]

Смысл минимума. Если преимущество положительно, вклад ограничен сверху
множителем ``1+ε``: «повышать вероятность хорошего действия полезно, но
не бесконечно за один заход». Если отрицательно — ограничен снизу ``1−ε``.
В обе стороны это означает: **шаг, уводящий политику дальше ε, не даёт
дополнительной выгоды**, и градиент по нему обнуляется.

Благодаря этому по одному роллауту можно пройти несколько эпох
мини-батчами — и именно поэтому PPO при том же бюджете шагов среды учится
заметно быстрее A2C.

Остальное совпадает с A2C: тот же критик, то же обобщённое преимущество
(GAE), тот же бонус за энтропию.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labrl.algos.a2c import ActOutput, explained_variance
from labrl.buffers.rollout import RolloutBatch
from labrl.nets.gaussian_policy import to_env_action


@dataclass
class PPOConfig:
    """Все гиперпараметры метода. «Магических чисел» внутри методов нет (8.6)."""

    #: Коэффициент дисконтирования, безразмерный, [0, 1).
    gamma: float = 0.99
    #: Параметр λ обобщённого преимущества (GAE), [0, 1].
    gae_lambda: float = 0.95
    #: Скорость обучения общего оптимизатора Adam.
    learning_rate: float = 3e-4
    #: Ширина обрезки ε отношения правдоподобий.
    clip_range: float = 0.2
    #: Сколько раз пройти по собранному роллауту.
    epochs: int = 4
    #: Размер мини-батча внутри эпохи, переходов.
    minibatch_size: int = 256
    #: Вес ошибки критика в общей функции потерь.
    value_coef: float = 0.5
    #: Вес бонуса за энтропию.
    entropy_coef: float = 0.005
    #: Максимальная норма градиента; 0 — не клипировать.
    max_grad_norm: float = 0.5
    #: Нормировать ли преимущества по мини-батчу.
    normalize_advantage: bool = True
    #: Порог приближённой KL, после которого эпохи прекращаются досрочно.
    #: 0 — не прекращать. Страховка на случай, когда обрезки не хватило.
    target_kl: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda должна быть в [0, 1], получено {self.gae_lambda}")
        if self.clip_range <= 0.0:
            raise ValueError(f"clip_range должен быть > 0, получено {self.clip_range}")
        if self.epochs <= 0:
            raise ValueError(f"epochs должно быть > 0, получено {self.epochs}")
        if self.minibatch_size <= 0:
            raise ValueError(f"minibatch_size должен быть > 0, получено {self.minibatch_size}")


class PPO:
    """PPO с обрезкой отношения правдоподобий, для непрерывных действий.

    Args:
        policy_net: актор — гауссова политика (см.
            :class:`labrl.nets.gaussian_policy.GaussianPolicyNetwork`).
            Архитектуру задаёт пользователь (требование 8.4).
        value_net: критик ``(B, obs_dim) -> (B, 1)``.
        cfg: гиперпараметры.
        device: устройство вычислений.
        seed: сид перемешивания мини-батчей.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        cfg: PPOConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)

        self.policy_net = policy_net.to(self.device)
        self.value_net = value_net.to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.cfg.learning_rate,
        )
        self._rng = np.random.default_rng(seed)
        self.updates = 0

        # Энтропия гауссовой политики зависит только от σ и наблюдения
        # не требует; энтропия категориальной вычисляется по логитам и без
        # наблюдения не определена. Правило PPO от этого не меняется, поэтому
        # различие снимается здесь — один раз, по сигнатуре метода, — а не
        # копированием update() в отдельный класс.
        self._entropy_needs_obs = bool(
            inspect.signature(self.policy_net.entropy).parameters
        )

        # log σ есть только у гауссовой политики: у категориальной ширина
        # разведки выражается энтропией, а не отдельным параметром. Тег
        # `Custom/Log Std` при этом остаётся в схеме и пишется нулём —
        # состав метрик не должен зависеть от типа политики.
        self._has_log_std = hasattr(self.policy_net, "clamped_log_std")

    # --- взаимодействие со средой ---------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> ActOutput:
        """Сэмплирует действие из политики. См. :meth:`labrl.algos.a2c.A2C.act`."""
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
        """Среднее политики, приведённое к [-1, 1] — то же, что выдаёт граф ONNX."""
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return to_env_action(self.policy_net(tensor).cpu().numpy())

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        """``V(s)`` для батча наблюдений — нужен для бутстрэппинга в GAE."""
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.value_net(tensor).squeeze(-1).cpu().numpy()

    # --- обучение --------------------------------------------------------

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """Единственное место, где меняются параметры (требование 8.7).

        Несколько эпох мини-батчами по одному роллауту. Данные при этом
        стареют — на то и обрезка: она гасит вклад переходов, для которых
        новая политика ушла от старой дальше чем на ``clip_range``.
        """
        size = len(batch)
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=self.device)
        old_log_prob = torch.as_tensor(batch.log_prob, dtype=torch.float32, device=self.device)
        advantage_all = torch.as_tensor(batch.advantage, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)

        stats: dict[str, float] = {}
        clip_fractions: list[float] = []
        epochs_done = 0
        stop = False

        for _ in range(self.cfg.epochs):
            order = self._rng.permutation(size)
            for start in range(0, size, self.cfg.minibatch_size):
                index = torch.as_tensor(
                    order[start : start + self.cfg.minibatch_size], device=self.device
                )
                if index.numel() < 2:
                    continue

                advantage = advantage_all[index]
                if self.cfg.normalize_advantage:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                log_prob = self.policy_net.log_prob(obs[index], action[index])
                ratio = torch.exp(log_prob - old_log_prob[index])

                # Ядро PPO: минимум из «как есть» и «с обрезанным отношением».
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantage
                policy_loss = -torch.min(unclipped, clipped).mean()

                value = self.value_net(obs[index]).squeeze(-1)
                value_loss = F.mse_loss(value, returns[index])
                entropy = (self.policy_net.entropy(obs[index]) if self._entropy_needs_obs
                           else self.policy_net.entropy())

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
                    # Оценка KL по Schulman: (r − 1) − log r. Она неотрицательна
                    # и точнее наивной −log r при малых отклонениях.
                    approx_kl = ((ratio - 1.0) - (log_prob - old_log_prob[index])).mean()
                    clip_fractions.append(
                        float(((ratio - 1.0).abs() > self.cfg.clip_range).float().mean().item())
                    )

                stats = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                    "grad_norm": float(grad_norm),
                    "log_std": (float(self.policy_net.clamped_log_std().mean().item())
                                if self._has_log_std else 0.0),
                }

                if self.cfg.target_kl > 0.0 and stats["approx_kl"] > self.cfg.target_kl:
                    stop = True
                    break

            epochs_done += 1
            if stop:
                break

        with torch.no_grad():
            predicted = self.value_net(obs).squeeze(-1).cpu().numpy()

        stats.update(
            {
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "explained_variance": explained_variance(predicted, batch.returns),
                "epochs_done": float(epochs_done),
                "updates": float(self.updates),
            }
        )
        return stats

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
        """Модуль, экспортируемый в ONNX: актор. Критик в Unity не нужен."""
        return self.policy_net.eval()
