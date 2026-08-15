"""DQN — Deep Q-Network (Mnih et al., 2015). Реализация с нуля на PyTorch.

Идея в двух предложениях. Табличный Q-learning хранит Q(s, a) для каждого
состояния; когда состояний слишком много (или они непрерывны), таблицу заменяют
сетью Q_θ(s, a). Всё остальное — то же обновление к цели Беллмана, но с двумя
добавками, без которых обучение расходится.

**Добавка 1: буфер воспроизведения.** Последовательные переходы коррелированы,
а градиентный спуск требует приближённо независимой выборки. Буфер перемешивает
опыт во времени (см. :mod:`labrl.buffers.replay`).

**Добавка 2: целевая сеть.** Цель обновления сама зависит от обучаемых весов:
``y = r + γ·max_a' Q_θ(s', a')``. Обучать сеть на цели, которая меняется вместе
с сетью, — рецепт расходимости: она «гонится за собственным хвостом». Поэтому
цель считается по **замороженной копии** Q_θ⁻, которая обновляется раз
в ``target_update_interval`` шагов.

Итоговое правило::

    y      = r + γ · (1 − terminated) · max_a' Q_θ⁻(s', a')
    L(θ)   = SmoothL1( Q_θ(s, a) , y )

Про ``(1 − terminated)``. Будущее обнуляется **только** при истинном завершении
эпизода. Обрыв по ``MaxStep`` (``truncated``) — не терминальное состояние:
там будущее есть, и бутстрэппинг обязан выполняться. Смешать эти случаи —
значит научить агента считать нехватку времени катастрофой; обучение при этом
не падает, оно тихо портится.

Про функцию потерь. Берётся Smooth L1 (Huber), а не MSE: TD-ошибка в начале
обучения бывает большой, и квадратичная функция превращает её в градиент,
сносящий сеть. Huber ограничивает влияние выбросов, оставаясь квадратичной
вблизи нуля.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from labrl.buffers.replay import Batch


@dataclass
class DQNConfig:
    """Все гиперпараметры метода. «Магических чисел» внутри методов нет (8.6)."""

    #: Коэффициент дисконтирования, безразмерный, [0, 1).
    gamma: float = 0.99
    #: Скорость обучения оптимизатора Adam.
    learning_rate: float = 3e-4
    #: Размер батча, переходов.
    batch_size: int = 128
    #: Раз во сколько **обновлений** копировать веса в целевую сеть.
    target_update_interval: int = 500
    #: Максимальная норма градиента; 0 — не клипировать.
    max_grad_norm: float = 10.0
    #: Использовать Double DQN: выбор действия онлайн-сетью, оценка — целевой.
    #: Обычный DQN систематически переоценивает Q, потому что max по зашумлённым
    #: оценкам смещён вверх; разделение выбора и оценки это смещение снимает
    #: (van Hasselt et al., 2016). Стоит одну лишнюю прямую прогонку.
    double_dqn: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma должна быть в [0, 1), получено {self.gamma}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate должен быть > 0, получено {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size должен быть > 0, получено {self.batch_size}")
        if self.target_update_interval <= 0:
            raise ValueError(
                f"target_update_interval должен быть > 0, получено {self.target_update_interval}"
            )


class DQN:
    """DQN с целевой сетью и опциональным Double DQN.

    Args:
        q_net: обучаемая Q-сеть. Любой ``nn.Module``, удовлетворяющий протоколу
            :class:`labrl.nets.protocols.QNetwork`: ``(B, obs_dim) -> (B, num_actions)``.
            Архитектуру задаёт пользователь — алгоритм её не знает (требование 8.4).
        target_net: сеть **той же архитектуры** для вычисления цели. Её веса
            перезаписываются из ``q_net``; собственного обучения у неё нет.
        cfg: гиперпараметры.
        device: устройство вычислений.
    """

    def __init__(
        self,
        q_net: nn.Module,
        target_net: nn.Module,
        cfg: DQNConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or DQNConfig()
        self.device = torch.device(device)

        self.q_net = q_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.sync_target()
        # Целевая сеть не обучается: её градиенты не нужны и только тратят память.
        for param in self.target_net.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        self.updates = 0

    # --- взаимодействие со средой ---------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        """ε-жадный выбор действия для батча наблюдений.

        Args:
            obs: ``(B, obs_dim)``.
            epsilon: вероятность случайного действия.
            rng: генератор — передаётся явно ради воспроизводимости.

        Returns:
            ``(B,)`` индексы действий.
        """
        greedy = self.greedy_action(obs)
        if epsilon <= 0.0:
            return greedy
        explore = rng.random(greedy.shape[0]) < epsilon
        random_actions = rng.integers(0, self.num_actions, size=greedy.shape[0])
        return np.where(explore, random_actions, greedy)

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> np.ndarray:
        """Жадное действие ``argmax_a Q(s, a)`` — итоговая политика метода."""
        was_training = self.q_net.training
        self.q_net.eval()
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        actions = self.q_net(tensor).argmax(dim=1).cpu().numpy()
        if was_training:
            self.q_net.train()
        return actions

    @property
    def num_actions(self) -> int:
        """Число действий определяется по выходу сети, а не задаётся отдельно."""
        if not hasattr(self, "_num_actions"):
            with torch.no_grad():
                probe = torch.zeros(1, self._infer_obs_dim(), device=self.device)
                self._num_actions = int(self.q_net(probe).shape[1])
        return self._num_actions

    def _infer_obs_dim(self) -> int:
        for module in self.q_net.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise ValueError(
            "не удалось определить размерность входа Q-сети: "
            "передайте сеть, начинающуюся с nn.Linear, либо задайте num_actions явно"
        )

    # --- обучение --------------------------------------------------------

    def update(self, batch: Batch) -> dict[str, float]:
        """Единственное место, где меняются параметры (требование 8.7).

        Returns:
            Метрики шага обучения для TensorBoard.
        """
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device)
        reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.bool, device=self.device)

        # Q(s, a) — значение выбранного действия.
        q_values = self.q_net(obs)
        q_selected = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # Double DQN: ДЕЙСТВИЕ выбирает онлайн-сеть, ОЦЕНИВАЕТ — целевая.
                best_action = self.q_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, best_action).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values

            # Будущее обнуляется только при истинном завершении. При truncated
            # бутстрэппинг выполняется — иначе обрыв по времени учил бы агента,
            # что времени всегда «не хватает катастрофически».
            target = reward + self.cfg.gamma * next_q * (~terminated).float()

        loss = F.smooth_l1_loss(q_selected, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(),
            self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf"),
        )
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.cfg.target_update_interval == 0:
            self.sync_target()

        with torch.no_grad():
            td_error = (target - q_selected).abs().mean().item()

        return {
            "loss": float(loss.item()),
            "td_error_abs": float(td_error),
            "q_mean": float(q_selected.mean().item()),
            "q_max": float(q_values.max().item()),
            "grad_norm": float(grad_norm),
            "updates": float(self.updates),
        }

    def sync_target(self) -> None:
        """Копирует веса онлайн-сети в целевую (жёсткое обновление)."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # --- сохранение и экспорт -------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))

    def policy_module(self) -> nn.Module:
        """Модуль, экспортируемый в ONNX.

        Это сама Q-сеть: обёртка экспорта (`labrl.export.onnx_export`) со
        стратегией ``greedy`` берёт от неё ``argmax`` — ровно ту политику,
        которую оценивает :meth:`greedy_action`. Никакой отдельной «политики»
        у DQN нет, и подменять её приближением не требуется.
        """
        return self.q_net.eval()
