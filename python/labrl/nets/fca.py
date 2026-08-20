"""Слой понятий FCA внутри графа политики.

Идея
----
:mod:`labrl.utils.fca` находит в опыте агента формальные понятия — устойчивые
сочетания бинарных признаков вида «есть ключ **и** я справа от стены».
Этот модуль превращает найденные понятия в **слой сети**: для каждого понятия
слой отвечает, содержит ли текущее состояние всё его содержание.

Зачем это слой, а не препроцессинг в Python
--------------------------------------------
Требование 10.7 инструкции прямое: всё, что преобразует наблюдение, обязано
попасть **в граф ONNX**. Признаки, посчитанные в Python и не уехавшие
в Unity, — гарантированный источник расхождения обучения и инференса,
причём молчаливого: сеть в Unity получит другой вход и будет работать,
просто плохо.

Поэтому проверка принадлежности понятию выражена так, чтобы её принимал
`torch.onnx.export` при opset 9::

    a       = σ( k · (obs − θ) )              бинаризация признаков
    score_k = a · intent_kᵀ − (|intent_k| − ½)
    m_k     = σ( β · score_k )                принадлежность понятию

Обе строки — это ``Linear`` с **фиксированными** весами плюс сигмоида,
то есть `MatMul`, `Add` и `Sigmoid`. Ничего сверх opset 9 не требуется.

Почему мягкая проверка, а не жёсткая
------------------------------------
Жёсткое «все признаки на месте» — это ``min`` по признакам содержания,
и производной у него нет почти везде. Мягкая версия с большим ``β``
численно неотличима от жёсткой (при ``β = 20`` разница меньше 1e-8
на целых входах), но дифференцируема, и градиент проходит к слоям
до неё, если пользователь захочет обучать и бинаризацию тоже.

Смысл смещения ``|intent| − ½``. Скалярное произведение бинарного вектора
признаков и бинарного содержания равно числу совпавших признаков. Оно равно
``|intent|`` тогда и только тогда, когда есть все; на единицу меньше —
если не хватает хотя бы одного. Порог ровно посередине разделяет эти случаи
максимально далеко от обоих.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from labrl.utils.fca import Concept, FormalContext

#: Имена бинарных признаков `E11_Research`. Порядок обязан совпадать
#: с первыми семью наблюдениями `KeyDoorAgent.CollectObservations`:
#: слой индексирует их напрямую, и перестановка сломала бы смысл каждого
#: понятия, не выдав ошибки.
KEYDOOR_ATTRIBUTES: tuple[str, ...] = (
    "есть ключ",
    "дверь открыта",
    "агент справа от стены",
    "агент выше линии двери",
    "агент на линии двери",
    "ключ ближе двери",
    "цель ближе двери",
)


def binarize_threshold(obs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Бинаризация наблюдений по порогу. ``(N, d) -> (N, d)`` из нулей и единиц.

    Для `E11_Research` порог не нужен: первые семь признаков уже бинарны,
    и функция лишь приводит тип. Она существует ради сред, где признаки
    непрерывны и контекст приходится строить, разрезая их порогом.
    """
    return (np.asarray(obs, dtype=np.float32) >= threshold).astype(np.float32)


class FCAConceptLayer(nn.Module):
    """``(B, num_attributes) -> (B, num_concepts)`` — принадлежность понятиям.

    Args:
        concepts: понятия, найденные :meth:`labrl.utils.fca.FormalContext.concepts`.
        num_attributes: число бинарных признаков контекста.
        threshold: порог бинаризации признаков.
        binarize_sharpness: крутизна ``k`` сигмоиды бинаризации.
        concept_sharpness: крутизна ``β`` сигмоиды принадлежности.

    Веса слоя **не обучаются**: понятия найдены по данным один раз и дальше
    фиксированы (`ENV_SPEC.md` среды `E11`, §7, TS11-3). Понятия,
    перестраиваемые по ходу обучения, меняли бы размерность входа политики
    прямо во время обучения.

    Понятие с пустым содержанием отбрасывается: оно истинно всегда
    и даёт константный признак.
    """

    def __init__(
        self,
        concepts: Sequence[Concept],
        num_attributes: int,
        threshold: float = 0.5,
        binarize_sharpness: float = 20.0,
        concept_sharpness: float = 20.0,
    ) -> None:
        super().__init__()

        kept = [c for c in concepts if len(c.intent) > 0]
        if not kept:
            raise ValueError(
                "нет ни одного понятия с непустым содержанием: контекст либо "
                "пуст, либо min_support отсёк всё. Понятие с пустым содержанием "
                "истинно всегда и признаком быть не может"
            )

        self.num_attributes = int(num_attributes)
        self.concepts = list(kept)
        self.threshold = float(threshold)
        self.binarize_sharpness = float(binarize_sharpness)
        self.concept_sharpness = float(concept_sharpness)

        intents = np.zeros((len(kept), self.num_attributes), dtype=np.float32)
        sizes = np.zeros(len(kept), dtype=np.float32)
        for row, concept in enumerate(kept):
            for attribute in concept.intent:
                if attribute >= self.num_attributes:
                    raise ValueError(
                        f"понятие ссылается на признак {attribute}, а признаков "
                        f"всего {self.num_attributes}"
                    )
                intents[row, attribute] = 1.0
            sizes[row] = len(concept.intent)

        # register_buffer, а не Parameter: веса зафиксированы данными
        # и обучению не подлежат, но обязаны попасть в state_dict
        # и в граф ONNX как константы.
        self.register_buffer("intents", torch.from_numpy(intents))
        self.register_buffer("intent_sizes", torch.from_numpy(sizes))

    @property
    def num_concepts(self) -> int:
        return len(self.concepts)

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        """Args: ``attributes`` ``(B, num_attributes)``. Returns: ``(B, num_concepts)``."""
        # Бинаризация: при целых 0/1 на входе и k = 20 сигмоида даёт
        # 0.99999 и 0.00001 — численно это те же ноль и единица.
        binary = torch.sigmoid(self.binarize_sharpness * (attributes - self.threshold))
        # Число совпавших признаков минус (|intent| − ½): положительно
        # тогда и только тогда, когда совпали все.
        matched = binary @ self.intents.t() - (self.intent_sizes - 0.5)
        return torch.sigmoid(self.concept_sharpness * matched)

    @torch.no_grad()
    def membership_exact(self, attributes: np.ndarray) -> np.ndarray:
        """Жёсткая принадлежность — эталон для проверки мягкой версии.

        В графе не используется: нужна тестам, которые сверяют, что мягкая
        проверка на целых входах совпадает с точной.
        """
        binary = (np.asarray(attributes, dtype=np.float32) >= self.threshold)
        intents = self.intents.cpu().numpy() > 0.5
        return np.array([
            [bool(binary[row][intents[k]].all()) for k in range(self.num_concepts)]
            for row in range(binary.shape[0])
        ], dtype=np.float32)

    def describe(self, names: Sequence[str] | None = None) -> str:
        """Список понятий слоя — для ячейки ноутбука."""
        lines = [f"понятий: {self.num_concepts} (признаков: {self.num_attributes})"]
        for index, concept in enumerate(self.concepts):
            lines.append(f"  {index:>2}. {concept.describe(names)}")
        return "\n".join(lines)


class FCAPolicy(nn.Module):
    """Политика с признаками понятий: ``(B, obs_dim) -> (B, sum(branches))``.

    Устройство входа::

        obs ──┬─ [:num_attributes] ─► FCAConceptLayer ─► понятия (B, K)
              └─ весь obs ───────────────────────────────► обычные признаки
                                    concat ─► MLP ─► логиты

    Наблюдение подаётся **целиком**, а не только его непрерывная часть:
    понятия добавляются к признакам, а не заменяют их. Замена была бы
    сильным утверждением — что вся нужная информация выражается понятиями, —
    а проверяется здесь противоположное, более скромное: помогают ли они.

    Args:
        obs_dim: полная размерность наблюдения.
        branches: размеры дискретных веток действия.
        concept_layer: слой понятий (:class:`FCAConceptLayer`).
        hidden_sizes: скрытые слои.
        activation: имя активации из :data:`labrl.nets.mlp.ACTIVATIONS`.

    Интерфейс совпадает с
    :class:`labrl.nets.categorical_policy.MultiBranchCategoricalPolicy`,
    поэтому политика подставляется в `PPODiscrete` без единой правки
    алгоритма — в этом и смысл протоколов сетей (требование 8.4).
    """

    def __init__(
        self,
        obs_dim: int,
        branches: Sequence[int],
        concept_layer: FCAConceptLayer,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        from labrl.nets.mlp import ACTIVATIONS, build_mlp

        if activation not in ACTIVATIONS:
            raise ValueError(f"activation должна быть одной из {sorted(ACTIVATIONS)}")
        if concept_layer.num_attributes > obs_dim:
            raise ValueError(
                f"слой понятий читает {concept_layer.num_attributes} признаков, "
                f"а наблюдение имеет размерность {obs_dim}"
            )

        self.obs_dim = int(obs_dim)
        self.branches = tuple(int(b) for b in branches)
        self.concept_layer = concept_layer
        self.net = build_mlp(
            self.obs_dim + concept_layer.num_concepts,
            sum(self.branches), hidden_sizes, activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        attributes = obs[:, : self.concept_layer.num_attributes]
        concepts = self.concept_layer(attributes)
        return self.net(torch.cat([obs, concepts], dim=-1))

    # --- распределения: интерфейс MultiBranchCategoricalPolicy -----------

    def split_logits(self, logits: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(logits, list(self.branches), dim=-1))

    def distributions(self, obs: torch.Tensor) -> list[torch.distributions.Categorical]:
        return [
            torch.distributions.Categorical(logits=branch_logits)
            for branch_logits in self.split_logits(self(obs))
        ]

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(obs.shape[0], device=obs.device)
        for branch, distribution in enumerate(self.distributions(obs)):
            total = total + distribution.log_prob(action[:, branch])
        return total

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=obs.device)
        for distribution in self.distributions(obs):
            total = total + distribution.entropy().mean()
        return total

    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions: list[torch.Tensor] = []
        log_probs = torch.zeros(obs.shape[0], device=obs.device)
        for distribution in self.distributions(obs):
            action = distribution.sample()
            log_probs = log_probs + distribution.log_prob(action)
            actions.append(action.unsqueeze(-1))
        return torch.cat(actions, dim=-1), log_probs

    @torch.no_grad()
    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [branch.argmax(dim=-1, keepdim=True) for branch in self.split_logits(self(obs))],
            dim=-1,
        )


def build_concept_layer(
    observations: np.ndarray,
    num_attributes: int,
    min_support: float = 0.02,
    max_intent: int | None = None,
    attribute_names: Sequence[str] | None = None,
) -> tuple[FCAConceptLayer, FormalContext]:
    """Строит слой понятий по собранному опыту.

    Args:
        observations: ``(N, obs_dim)`` — наблюдения, собранные любой политикой.
        num_attributes: сколько первых признаков наблюдения бинарны.
        min_support: минимальная доля объектов в объёме понятия. Отсекает
            редкие сочетания: признак, встретившийся в трёх состояниях
            из десяти тысяч, — это шум опыта, а не структура задачи.
        max_intent: максимальное число признаков в содержании.
        attribute_names: имена признаков для отчётов.

    Returns:
        ``(слой, контекст)``. Контекст возвращается, чтобы ноутбук мог
        показать частоты признаков и импликации.
    """
    attributes = binarize_threshold(np.asarray(observations)[:, :num_attributes])
    context = FormalContext(attributes, attribute_names)
    concepts = context.concepts(min_support=min_support, max_intent=max_intent)
    return FCAConceptLayer(concepts, num_attributes), context
