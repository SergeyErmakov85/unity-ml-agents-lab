"""PPO для дискретных действий с несколькими ветками.

Это **не другой алгоритм**. Правило обновления, обрезка отношения
правдоподобий, GAE и функция потерь — те же, что в :mod:`labrl.algos.ppo`,
и метод ``update()`` наследуется оттуда без изменений: дублировать его
значило бы завести две реализации одной математики, которые рано или поздно
разойдутся (требование 8.7).

Различие ровно в двух местах, и оба живут в **распределении политики**,
а не в алгоритме:

============================  ============================  ==========================
что                           непрерывный случай             дискретный случай
============================  ============================  ==========================
распределение                 гауссиана ``N(μ(s), σ)``       произведение категориальных
что возвращает сеть           среднее ``μ(s)``               логиты всех веток подряд
как сэмплируется действие     ``μ + σ·ε``, ``ε ~ N(0,1)``    ``multinomial(softmax)`` по ветке
энтропия                      зависит только от ``σ``        зависит от наблюдения
действие для среды            float в ``[-1, 1]``            целые индексы ``(B, ветки)``
============================  ============================  ==========================

Последняя строка объясняет, почему у дискретного случая нет разделения
на «сырое» и «приведённое» действие: индекс ветки уже и есть то, что уходит
в среду, и приводить его к диапазону не нужно.

Предпоследняя строка — единственное, что потребовало правки в базовом классе:
``PPO.__init__`` определяет по сигнатуре ``policy_net.entropy``, нужно ли
ей наблюдение. Гауссова политика отвечает «нет», категориальная — «да».
"""

from __future__ import annotations

import numpy as np
import torch

from labrl.algos.a2c import ActOutput
from labrl.algos.ppo import PPO


class PPODiscrete(PPO):
    """PPO с категориальной политикой из нескольких независимых веток.

    Args:
        policy_net: актор, возвращающий ``(B, sum(branches))`` логитов и
            реализующий ``log_prob(obs, action)``, ``entropy(obs)``,
            ``sample(obs)``, ``greedy(obs)`` — например
            :class:`labrl.nets.categorical_policy.MultiBranchCategoricalPolicy`.
            Архитектуру задаёт пользователь (требование 8.4).
        value_net: критик ``(B, obs_dim) -> (B, 1)``.
        cfg: гиперпараметры — тот же :class:`labrl.algos.ppo.PPOConfig`.
        device: устройство вычислений.
        seed: сид перемешивания мини-батчей.
    """

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator) -> ActOutput:
        """Сэмплирует действие из категориальной политики.

        Args:
            obs: ``(B, obs_dim)``.
            rng: генератор **не используется**: сэмплирование категориального
                распределения делает PyTorch (``torch.multinomial``), тот же,
                что работает внутри графа ONNX. Заводить для них разные
                источники случайности значило бы гарантированно получить
                разное поведение Python и Unity. Аргумент оставлен, чтобы
                сигнатура совпадала с непрерывным случаем и цикл обучения
                (`labrl.train.onpolicy`) не различал их.

        Returns:
            :class:`labrl.algos.a2c.ActOutput`, где ``env_action`` и
            ``raw_action`` — один и тот же массив индексов ``(B, ветки)``.
        """
        del rng  # см. docstring

        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        action, log_prob = self.policy_net.sample(tensor)
        value = self.value_net(tensor).squeeze(-1)

        action_np = action.cpu().numpy().astype(np.int64)
        return ActOutput(
            # Индекс ветки — уже готовое действие среды; «сырого» варианта,
            # который надо было бы приводить к диапазону, здесь нет.
            env_action=action_np,
            raw_action=action_np,
            log_prob=log_prob.cpu().numpy(),
            value=value.cpu().numpy(),
        )

    @torch.no_grad()
    def deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Наиболее вероятное действие каждой ветки.

        То же, что отдаёт выход ``deterministic_discrete_actions`` графа ONNX
        (`docs/04_ONNX_CONTRACT.md`, §2), — и это не совпадение, а требование:
        оценка в Python и инференс в Unity обязаны выбирать одно и то же.
        """
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.policy_net.greedy(tensor).cpu().numpy().astype(np.int64)
