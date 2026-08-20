"""BC — Behavioral Cloning. Реализация с нуля.

Самый простой метод в лаборатории и единственный, который **не является
обучением с подкреплением**. Ни награды, ни ценности, ни исследования:
есть пары «наблюдение → действие» от эксперта, и задача сводится
к обычной классификации.

    L(θ) = − E_{(s,a) ~ D_эксперта} [ log π_θ(a | s) ]

Для дискретных действий это в точности перекрёстная энтропия, где «классом»
служит индекс действия эксперта.

Почему это работает мгновенно
-----------------------------
RL добывает сигнал взаимодействием: чтобы узнать, что ход был хорош, нужно
дойти до награды. BC получает правильный ответ на каждый пример сразу.
На `E10_Imitation` сотня демонстраций — это около трёх тысяч пар, и обучение
занимает секунды против сотен тысяч шагов среды у PPO.

Почему этого мало: сдвиг распределения
--------------------------------------
BC учится на состояниях, **которые посещал эксперт**. Как только политика
ошибётся и попадёт в состояние, которого в демонстрациях не было, её выход
там ничем не обусловлен — и следующая ошибка вероятнее предыдущей. Ошибки
накапливаются: Ross и Bagnell (2010) показали, что при вероятности ошибки ε
на шаге суммарная потеря растёт как ``O(ε·T²)``, а не ``O(ε·T)``.

Когда этого НЕ происходит — и почему `E10_Imitation` тому пример
-----------------------------------------------------------------
Сдвиг распределения нуждается в состояниях, куда политика попадает,
а эксперт — нет. В коридоре без развилок таких состояний почти нет:
эксперт проходит все 49 клеток маршрута, и политика, не делающая ошибок,
не выходит за пределы показанного.

Измерено на `E10_Imitation`: BC доходит до выхода в **100 %** эпизодов
после двух тысяч мини-батчей (18 секунд), тогда как случайная политика —
в 0.0 %. То есть на этой среде BC работает, и работает превосходно.

Это не опровержение теоремы Ross & Bagnell, а уточнение области её
применимости: **оценка ε·T² — верхняя граница, и достигается она только
там, где ошибка выводит агента в незнакомую область**. Чтобы увидеть сдвиг
распределения на практике, нужна среда, где у агента есть куда сойти
с траектории эксперта, — например с развилками, случайными помехами или
непрерывным управлением, где точное повторение невозможно в принципе.

Лекарства от сдвига два, и оба выходят за рамки BC: **DAgger** (спрашивать
эксперта в новых состояниях — требует эксперта в цикле обучения) и **GAIL**
(:mod:`labrl.algos.gail` — самому создавать сигнал в незнакомых состояниях).
На `E10_Imitation` GAIL нужен не поэтому, а ради другого свойства: он вообще
не пользуется функцией награды.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from labrl.envs.demos import Demonstrations


@dataclass
class BCConfig:
    """Все гиперпараметры метода. «Магических чисел» внутри методов нет (8.6)."""

    #: Скорость обучения Adam.
    learning_rate: float = 1e-3
    #: Размер мини-батча, пар.
    batch_size: int = 256
    #: Вес L2-регуляризации (weight decay Adam).
    weight_decay: float = 0.0
    #: Вес бонуса за энтропию. Ноль — чистая классификация.
    #: Положительный удерживает политику от полной уверенности: при переходе
    #: к GAIL или дообучению нулевая энтропия означает нулевую разведку.
    entropy_coef: float = 0.0
    #: Максимальная норма градиента; 0 — не клипировать.
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size должен быть > 0, получено {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate должен быть > 0, получено {self.learning_rate}")


class BC:
    """Обучение с учителем на демонстрациях эксперта.

    Args:
        policy_net: та же категориальная политика, что и у RL-методов
            (:class:`labrl.nets.categorical_policy.MultiBranchCategoricalPolicy`).
            Общая архитектура — не совпадение: BC часто используют
            как инициализацию для последующего RL, и сеть должна быть той же.
        cfg: гиперпараметры.
        device: устройство вычислений.
        seed: сид выборки мини-батчей.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        cfg: BCConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or BCConfig()
        self.device = torch.device(device)
        self.policy_net = policy_net.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self._rng = np.random.default_rng(seed)
        self.updates = 0

    # --- взаимодействие со средой ---------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Сэмплированное действие — для сбора опыта и оценки со стохастикой."""
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        action, _ = self.policy_net.sample(tensor)
        return action.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Наиболее вероятное действие — то же, что даёт граф ONNX."""
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.policy_net.greedy(tensor).cpu().numpy().astype(np.int64)

    # --- обучение --------------------------------------------------------

    def update(self, batch: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        """Единственное место, где меняются параметры (требование 8.7).

        Args:
            batch: ``(obs (B, obs_dim), action (B, num_branches))`` — пары эксперта.
        """
        obs_np, action_np = batch
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=self.device)
        action = torch.as_tensor(np.asarray(action_np, dtype=np.int64), device=self.device)

        # log π(a_эксперта | s), просуммированный по веткам. Минус среднее —
        # это и есть перекрёстная энтропия для факторизованной политики.
        log_prob = self.policy_net.log_prob(obs, action)
        loss = -log_prob.mean()

        entropy = self.policy_net.entropy(obs)
        if self.cfg.entropy_coef > 0.0:
            loss = loss - self.cfg.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf"),
        )
        self.optimizer.step()
        self.updates += 1

        with torch.no_grad():
            predicted = self.policy_net.greedy(obs)
            accuracy = (predicted == action).all(dim=1).float().mean()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(-log_prob.mean().item()),
            "entropy": float(entropy.item()),
            "accuracy": float(accuracy.item()),
            "grad_norm": float(grad_norm),
            "updates": float(self.updates),
        }

    def fit(self, demos: Demonstrations, steps: int) -> list[dict[str, float]]:
        """Прогоняет ``steps`` мини-батчей. Возвращает историю статистик.

        Мини-батчи берутся **с возвращением**, а не проходом по эпохам:
        разница на наборе из тысяч пар незаметна, а код короче.
        """
        history: list[dict[str, float]] = []
        for _ in range(int(steps)):
            history.append(self.update(demos.sample(self.cfg.batch_size, self._rng)))
        return history

    @torch.no_grad()
    def accuracy(self, demos: Demonstrations, batch_size: int = 4096) -> float:
        """Доля пар, на которых политика выбирает то же действие, что эксперт.

        На **отложенной** части набора это единственная честная мера того,
        выучил ли BC правило эксперта, а не запомнил записи.

        Важно: высокая точность **не гарантирует** высокой доли пройденных
        эпизодов — см. про сдвиг распределения в описании модуля.
        """
        correct = 0
        for start in range(0, len(demos), batch_size):
            obs = torch.as_tensor(demos.obs[start : start + batch_size], device=self.device)
            action = torch.as_tensor(demos.action[start : start + batch_size], device=self.device)
            predicted = self.policy_net.greedy(obs)
            correct += int((predicted == action).all(dim=1).sum().item())
        return correct / max(len(demos), 1)

    def set_learning_rate(self, learning_rate: float) -> None:
        if learning_rate < 0.0:
            raise ValueError(f"learning_rate должен быть >= 0, получено {learning_rate}")
        self.cfg.learning_rate = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    # --- сохранение и экспорт -------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(state["policy_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))

    def policy_module(self) -> nn.Module:
        """Модуль, экспортируемый в ONNX. У BC это вся модель целиком."""
        return self.policy_net.eval()
