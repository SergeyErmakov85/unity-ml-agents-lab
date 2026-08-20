"""Формальный анализ понятий (FCA). Реализация с нуля.

Что это и зачем в лаборатории обучения с подкреплением
------------------------------------------------------
FCA — аппарат для извлечения иерархической структуры из **бинарной таблицы**
«объект — признак». Его основное применение в модуле `/fca-rl` курса —
анализ прогонов и архитектур; здесь он используется иначе: как способ
получить **составные признаки состояния** для политики.

Основные понятия
----------------
**Формальный контекст** — тройка ``(G, M, I)``: множество объектов ``G``,
множество признаков ``M`` и отношение инциденции ``I ⊆ G × M``
(«объект `g` обладает признаком `m`»). Практически это матрица нулей
и единиц.

**Операторы Галуа** связывают подмножества объектов и признаков::

    A' = { m ∈ M : ∀g ∈ A, (g, m) ∈ I }     общие признаки множества объектов
    B' = { g ∈ G : ∀m ∈ B, (g, m) ∈ I }     объекты, обладающие всеми признаками

**Формальное понятие** — пара ``(A, B)`` такая, что ``A' = B`` и ``B' = A``.
``A`` называется объёмом (extent), ``B`` — содержанием (intent). Иначе
говоря, понятие — это максимальный прямоугольник единиц в матрице:
добавить объект нельзя, не потеряв признак, и наоборот.

**Решётка понятий** — все понятия контекста, упорядоченные по включению
объёмов. Она и есть та иерархическая структура, ради которой всё затевается.

Почему именно замкнутые множества
---------------------------------
Соблазн «взять все комбинации признаков» приводит к ``2^|M|`` вариантам,
большинство из которых не встречается в данных. Замыкание оставляет
ровно те комбинации, которые **реально наблюдались вместе**, и склеивает
эквивалентные: если в данных «дверь открыта» всегда совпадает с «есть ключ»,
эти два признака попадут в одно понятие, а не в два.

Алгоритм
--------
Close-by-One (CbO, Kuznetsov, 1993) — перебор замкнутых множеств признаков
в лексикографическом порядке с отсечением дубликатов по канонической
проверке. Написан на булевых массивах NumPy: контекст здесь маленький
(тысячи объектов, единицы признаков), и городить битовые множества незачем.

Источник: Ganter & Wille, «Formal Concept Analysis: Mathematical
Foundations» (1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Concept:
    """Формальное понятие: объём и содержание.

    Attributes:
        extent: индексы объектов, обладающих всеми признаками содержания.
        intent: индексы признаков, общих для всех объектов объёма.
        support: доля объектов контекста в объёме — насколько понятие частое.
    """

    extent: tuple[int, ...]
    intent: tuple[int, ...]
    support: float

    def __len__(self) -> int:
        """Число признаков в содержании."""
        return len(self.intent)

    def describe(self, names: Sequence[str] | None = None) -> str:
        """Человекочитаемое содержание понятия."""
        if names is None:
            body = ", ".join(f"m{i}" for i in self.intent)
        else:
            body = ", ".join(names[i] for i in self.intent)
        return f"{{{body}}}  объектов: {len(self.extent)} ({self.support:.1%})"


class FormalContext:
    """Формальный контекст: матрица «объект × признак» из нулей и единиц.

    Args:
        incidence: ``(num_objects, num_attributes)`` — булев массив либо
            массив нулей и единиц.
        attribute_names: имена признаков. Нужны только для отчётов.

    Дубликаты строк **не удаляются**: доля объектов в объёме (`support`)
    должна отражать частоту состояния в опыте, а не число различных
    состояний.
    """

    def __init__(self, incidence: np.ndarray, attribute_names: Sequence[str] | None = None) -> None:
        incidence = np.asarray(incidence)
        if incidence.ndim != 2:
            raise ValueError(f"контекст должен быть матрицей (объекты × признаки), получено {incidence.shape}")

        unique = np.unique(incidence)
        if not np.all(np.isin(unique, (0, 1, True, False))):
            raise ValueError(
                "контекст обязан состоять из нулей и единиц; получены значения "
                f"{unique[:5].tolist()}. Непрерывные признаки нужно сначала "
                "бинаризовать (labrl.nets.fca.binarize_threshold)"
            )

        self.incidence = incidence.astype(bool)
        self.attribute_names = (list(attribute_names) if attribute_names is not None
                                else [f"m{i}" for i in range(self.num_attributes)])
        if len(self.attribute_names) != self.num_attributes:
            raise ValueError(
                f"имён признаков {len(self.attribute_names)}, а признаков {self.num_attributes}"
            )

    @property
    def num_objects(self) -> int:
        return int(self.incidence.shape[0])

    @property
    def num_attributes(self) -> int:
        return int(self.incidence.shape[1])

    # --- операторы Галуа -------------------------------------------------

    def objects_with(self, intent: Sequence[int] | np.ndarray) -> np.ndarray:
        """``B'`` — индексы объектов, обладающих **всеми** признаками ``B``.

        Пустое содержание даёт все объекты: условие «обладает всеми
        признаками пустого множества» выполнено тривиально.
        """
        intent = np.asarray(list(intent), dtype=np.int64)
        if intent.size == 0:
            return np.arange(self.num_objects)
        return np.flatnonzero(self.incidence[:, intent].all(axis=1))

    def common_attributes(self, extent: Sequence[int] | np.ndarray) -> np.ndarray:
        """``A'`` — индексы признаков, общих для **всех** объектов ``A``.

        Пустой объём даёт все признаки — двойственно предыдущему случаю.
        """
        extent = np.asarray(list(extent), dtype=np.int64)
        if extent.size == 0:
            return np.arange(self.num_attributes)
        return np.flatnonzero(self.incidence[extent, :].all(axis=0))

    def closure(self, intent: Sequence[int] | np.ndarray) -> np.ndarray:
        """``B''`` — замыкание содержания.

        Добавляет к ``B`` все признаки, которыми и без того обладают все
        объекты, у которых есть ``B``. Замыкание — это и есть содержание
        понятия, порождённого множеством ``B``.
        """
        return self.common_attributes(self.objects_with(intent))

    # --- перебор понятий -------------------------------------------------

    def concepts(self, min_support: float = 0.0, max_intent: int | None = None) -> list[Concept]:
        """Все формальные понятия контекста, отсортированные по частоте.

        Args:
            min_support: минимальная доля объектов в объёме. Отсекает редкие
                понятия: признак, встретившийся в трёх состояниях из десяти
                тысяч, — это шум опыта, а не структура задачи.
            max_intent: максимальное число признаков в содержании.
                ``None`` — без ограничения.

        Returns:
            Список понятий по убыванию частоты. Первым идёт понятие
            с пустым (или общим для всех) содержанием — оно охватывает
            весь контекст.

        Сложность в худшем случае экспоненциальна по числу признаков, и это
        свойство задачи, а не реализации: замкнутых множеств может быть
        ``2^|M|``. При семи признаках `E11_Research` это максимум 128 —
        перебор мгновенный. Для контекстов с десятками признаков нужен
        ``min_support``.
        """
        if not 0.0 <= min_support <= 1.0:
            raise ValueError(f"min_support должен быть в [0, 1], получено {min_support}")

        minimum = int(np.ceil(min_support * self.num_objects))
        found: list[Concept] = []
        seen: set[tuple[int, ...]] = set()

        for intent in self._close_by_one():
            if max_intent is not None and len(intent) > max_intent:
                continue
            key = tuple(int(i) for i in intent)
            if key in seen:
                continue
            extent = self.objects_with(intent)
            if len(extent) < minimum:
                continue
            seen.add(key)
            found.append(Concept(
                extent=tuple(int(g) for g in extent),
                intent=key,
                support=len(extent) / max(self.num_objects, 1),
            ))

        found.sort(key=lambda c: (-c.support, len(c.intent)))
        return found

    def _close_by_one(self) -> Iterator[np.ndarray]:
        """Перебор замкнутых множеств признаков (Close-by-One).

        Идея: идти по признакам в порядке возрастания индекса и расширять
        текущее замкнутое множество только признаками **больше** последнего
        добавленного. Каноническая проверка отбрасывает ветвь, если
        замыкание вернуло признак меньше текущего: такое множество уже
        было построено раньше по другому пути.
        """
        empty = self.closure(np.empty(0, dtype=np.int64))
        yield empty

        stack: list[tuple[np.ndarray, int]] = [(empty, -1)]
        while stack:
            intent, last = stack.pop()
            present = set(int(i) for i in intent)

            for attribute in range(last + 1, self.num_attributes):
                if attribute in present:
                    continue

                candidate = np.append(intent, attribute)
                closed = self.closure(candidate)

                # Каноническая проверка: новые признаки обязаны быть больше
                # добавленного. Иначе это множество уже встречалось.
                added = set(int(i) for i in closed) - present - {attribute}
                if any(a < attribute for a in added):
                    continue

                yield closed
                stack.append((closed, attribute))

    # --- отчёт -----------------------------------------------------------

    def describe(self) -> str:
        """Сводка контекста — для ячейки ноутбука."""
        density = self.incidence.mean() if self.incidence.size else 0.0
        lines = [
            f"объектов: {self.num_objects}, признаков: {self.num_attributes}, "
            f"плотность единиц: {density:.1%}",
            "частота признаков:",
        ]
        for index, name in enumerate(self.attribute_names):
            lines.append(f"  {name:<24} {self.incidence[:, index].mean():.1%}")
        return "\n".join(lines)


def implications(context: FormalContext, min_support: float = 0.0) -> list[tuple[tuple[int, ...], int]]:
    """Импликации вида «набор признаков ⇒ признак», выполненные в контексте.

    Возвращает пары ``(посылка, следствие)``. Импликация с одноэлементной
    посылкой и таким же следствием — это в точности эквивалентность двух
    признаков, и её появление в отчёте по `E11_Research` служит проверкой
    того, что метод работает: признаки «есть ключ» и «дверь открыта» там
    эквивалентны по построению среды, и FCA обязан это обнаружить сам.
    """
    result: list[tuple[tuple[int, ...], int]] = []
    for concept in context.concepts(min_support=min_support):
        if not concept.intent:
            continue
        for attribute in concept.intent:
            premise = tuple(a for a in concept.intent if a != attribute)
            if not premise:
                continue
            # Импликация нетривиальна, только если следствие не выводится
            # из посылки более коротким путём: замыкание посылки уже содержит
            # следствие тогда и только тогда, когда импликация верна.
            if attribute in set(int(i) for i in context.closure(premise)):
                result.append((premise, int(attribute)))
    return result
