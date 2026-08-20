"""Внимание над множеством сущностей переменного размера.

Зачем это понадобилось. Все критики лаборатории до `E08_SoccerArena` получали
на вход один вектор фиксированной длины. Централизованный критик команды так
устроить нельзя по двум причинам:

1. **Состав команды меняется.** Игрок может закончить эпизод раньше остальных
   (в футболе — нет, но в общем случае MA-POCA обязан это уметь: именно
   отсюда в названии метода «posthumous» — посмертное распределение заслуги).
   Склейка наблюдений в один вектор фиксирует число агентов навсегда.

2. **Порядок агентов не значит ничего.** «Игрок 0 и игрок 1» — то же самое,
   что «игрок 1 и игрок 0». Склейка же заставляет сеть выучить обе
   перестановки по отдельности.

Механизм, отвечающий обоим требованиям, — **внимание**: оно принимает
множество сущностей любого размера и по построению не зависит от их порядка,
если не добавлять позиционного кодирования (а мы не добавляем).

Реализация — Residual Self-Attention (RSA), как в статье MA-POCA
(Cohen et al., 2022) и в штатном тренере ML-Agents. Написана с нуля
на `torch.nn.Linear` и `softmax`, без `nn.MultiheadAttention`: учебный код
не должен прятать формулу внимания за библиотечным вызовом.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualSelfAttention(nn.Module):
    """Одно-головое self-attention с остаточной связью и маской сущностей.

    Args:
        embed_dim: размерность представления сущности.
        num_heads: число голов внимания.

    Формы:
        * вход ``entities``: ``(B, N, embed_dim)``;
        * вход ``mask``: ``(B, N)``, 1 — сущность существует, 0 — нет;
        * выход: ``(B, N, embed_dim)``.

    Формула одной головы::

        A = softmax( Q Kᵀ / √d  +  (mask − 1)·1e9 )
        out = A V

    Слагаемое ``(mask − 1)·1e9`` обнуляет внимание к несуществующим
    сущностям: у них показатель экспоненты уходит в −1e9, и после softmax
    их вес неотличим от нуля. Использовать ``-inf`` нельзя — если у сущности
    замаскированы **все** соседи, softmax по строке из −inf даёт NaN.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) должна делиться на num_heads ({num_heads})"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.out = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, entities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, count, _ = entities.shape

        normed = self.norm(entities)

        def split(x: torch.Tensor) -> torch.Tensor:
            # (B, N, E) -> (B, heads, N, head_dim)
            return x.view(batch, count, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(self.query(normed))
        k = split(self.key(normed))
        v = split(self.value(normed))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (B, N) -> (B, 1, 1, N): маскируется столбец, то есть сущность,
        # НА которую смотрят. Строки замаскированных сущностей считаются тоже,
        # но их результат отбрасывается при пулинге (см. masked_mean).
        scores = scores + (mask[:, None, None, :] - 1.0) * 1e9

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch, count, self.embed_dim)

        # Остаточная связь: внимание уточняет представление сущности,
        # а не заменяет его. Без неё градиент к энкодеру сущности идёт
        # только через softmax и на старте обучения почти не проходит.
        return entities + self.out(context)


def masked_mean(entities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Среднее по существующим сущностям. ``(B, N, E)``, ``(B, N)`` -> ``(B, E)``.

    Пулинг обязан быть симметричным (среднее, максимум, сумма), иначе
    независимость от порядка агентов, ради которой заведено внимание,
    теряется на последнем шаге.

    Среднее, а не сумма: сумма растёт с числом агентов, и критик, обученный
    на команде из двух, выдавал бы вдвое меньшее значение для команды
    из четырёх — то есть менял бы масштаб ценности при смене состава.
    """
    weights = mask.unsqueeze(-1)
    total = (entities * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return total / count


class EntityEncoder(nn.Module):
    """Кодировщик одной сущности в общее пространство представлений.

    Args:
        input_dim: размерность описания сущности.
        embed_dim: размерность представления, общая для всех типов сущностей.
        hidden_dim: ширина скрытого слоя.

    Разные типы сущностей (наблюдение агента; пара «наблюдение + действие»)
    имеют **разные** входные размерности, но обязаны попасть в одно
    пространство: иначе внимание не сможет сравнивать их между собой.
    Поэтому у каждого типа свой кодировщик и общий ``embed_dim``.
    """

    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(embed_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
