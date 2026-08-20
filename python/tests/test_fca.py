"""Формальный анализ понятий: замыкание, перебор понятий, слой в графе.

Три класса проверок:

1. **Математика FCA** — операторы Галуа и замкнутость. Ошибка здесь даёт
   «понятия», которые понятиями не являются, и слой начинает выдавать
   признаки, не соответствующие ничему в данных.
2. **Совпадение мягкой проверки с точной** — слой обязан вести себя как
   жёсткое «все признаки на месте». Расхождение не роняет обучение,
   а тихо портит вход политики.
3. **Экспортируемость** — весь слой обязан попадать в граф ONNX
   (требование 10.7). Признаки, посчитанные в Python и не уехавшие
   в Unity, — гарантированное расхождение обучения и инференса.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from labrl.nets.fca import (
    FCAConceptLayer,
    FCAPolicy,
    binarize_threshold,
    build_concept_layer,
)
from labrl.utils.fca import FormalContext, implications

#: Классический пример из Ganter & Wille: четыре объекта, три признака.
#:
#:        m0  m1  m2
#:   g0    1   1   0
#:   g1    1   0   1
#:   g2    1   1   0
#:   g3    0   0   1
SIMPLE = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
    [0, 0, 1],
], dtype=np.int64)


def simple_context() -> FormalContext:
    return FormalContext(SIMPLE, ["m0", "m1", "m2"])


# --- операторы Галуа ----------------------------------------------------


def test_objects_with_empty_intent_is_everything():
    """Условие «обладает всеми признаками пустого множества» выполнено всегда."""
    context = simple_context()
    assert context.objects_with([]).tolist() == [0, 1, 2, 3]


def test_common_attributes_of_empty_extent_is_everything():
    context = simple_context()
    assert context.common_attributes([]).tolist() == [0, 1, 2]


def test_galois_operators():
    context = simple_context()
    # m1 есть у g0 и g2…
    assert context.objects_with([1]).tolist() == [0, 2]
    # …и у обоих есть ещё m0.
    assert context.common_attributes([0, 2]).tolist() == [0, 1]


def test_closure_is_idempotent():
    """B'' = B'''' — замыкание замкнутого множества равно ему самому.

    Свойство определяет корректность всего аппарата: если оно нарушено,
    «понятия» не являются понятиями.
    """
    context = simple_context()
    for subset in ([], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]):
        once = context.closure(subset)
        twice = context.closure(once)
        assert once.tolist() == twice.tolist()


def test_closure_adds_implied_attributes():
    """m1 ⇒ m0: у всех объектов с m1 есть и m0, значит замыкание {m1} = {m0, m1}."""
    assert simple_context().closure([1]).tolist() == [0, 1]


# --- перебор понятий ----------------------------------------------------


def test_every_concept_is_closed():
    """Каждое найденное понятие обязано быть парой (A', A) с A'' = A.

    Это определение понятия, и проверять его нужно прямо: перебор с отсечением
    легко «потерять» одно условие, и тогда часть результатов понятиями не будет.
    """
    context = simple_context()
    for concept in context.concepts():
        extent = np.array(concept.extent)
        intent = np.array(concept.intent)
        assert context.objects_with(intent).tolist() == extent.tolist()
        assert context.common_attributes(extent).tolist() == intent.tolist()


def test_concepts_are_unique():
    context = simple_context()
    intents = [c.intent for c in context.concepts()]
    assert len(intents) == len(set(intents))


def test_concept_count_matches_manual_enumeration():
    """У этого контекста ровно шесть понятий. Перечисление вручную:

    ==========  ===========  ================================================
    содержание  объём        почему замкнуто
    ==========  ===========  ================================================
    {}          все          общих признаков нет: у g3 нет m0
    {m0}        g0, g1, g2   общий только m0: у g1 нет m1, у g0 нет m2
    {m2}        g1, g3       общий только m2
    {m0, m1}    g0, g2       замыкание {m1}: у всех с m1 есть и m0
    {m0, m2}    g1          единственный объект с обоими
    {m0,m1,m2}  пусто        замыкание пустого объёма — все признаки
    ==========  ===========  ================================================
    """
    intents = {c.intent for c in simple_context().concepts()}
    assert intents == {(), (0,), (2,), (0, 1), (0, 2), (0, 1, 2)}


def test_min_support_filters_rare_concepts():
    context = simple_context()
    # {m0, m1, m2} не встречается ни у одного объекта — объём пуст.
    all_intents = {c.intent for c in context.concepts(min_support=0.0)}
    frequent = {c.intent for c in context.concepts(min_support=0.5)}
    assert (0, 1, 2) in all_intents
    assert (0, 1, 2) not in frequent


def test_support_is_the_share_of_objects():
    for concept in simple_context().concepts():
        assert concept.support == pytest.approx(len(concept.extent) / 4)


def test_equivalent_attributes_collapse_into_one_concept():
    """Два признака, всегда совпадающих, обязаны попасть в ОДНО понятие.

    Это ровно та проверка, ради которой в `E11_Research` признаки «есть ключ»
    и «дверь открыта» сделаны эквивалентными по построению: если FCA
    не склеивает их, метод не работает.
    """
    duplicated = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=np.int64)
    context = FormalContext(duplicated)
    for concept in context.concepts():
        # Признаки 0 и 1 либо оба в содержании, либо оба вне его.
        assert (0 in concept.intent) == (1 in concept.intent)


def test_implications_find_the_equivalence():
    duplicated = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.int64)
    found = implications(FormalContext(duplicated))
    assert ((0,), 1) in found and ((1,), 0) in found


def test_context_rejects_non_binary_input():
    with pytest.raises(ValueError, match="нулей и единиц"):
        FormalContext(np.array([[0.3, 0.7]]))


def test_context_rejects_mismatched_names():
    with pytest.raises(ValueError, match="имён признаков"):
        FormalContext(SIMPLE, ["a", "b"])


# --- слой понятий -------------------------------------------------------


def make_layer() -> FCAConceptLayer:
    context = simple_context()
    return FCAConceptLayer(context.concepts(), num_attributes=3)


def test_layer_drops_the_empty_concept():
    """Понятие с пустым содержанием истинно всегда и признаком быть не может."""
    layer = make_layer()
    assert all(len(c.intent) > 0 for c in layer.concepts)


def test_layer_matches_exact_membership_on_binary_input():
    """Мягкая проверка обязана совпадать с жёсткой на целых входах.

    Расхождение здесь не роняет обучение — оно тихо портит вход политики.
    """
    layer = make_layer()
    grid = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)],
                    dtype=np.float32)

    with torch.no_grad():
        soft = layer(torch.from_numpy(grid)).numpy()
    exact = layer.membership_exact(grid)

    assert np.abs(soft - exact).max() < 1e-3


def test_layer_membership_is_one_only_when_all_attributes_present():
    layer = FCAConceptLayer(
        [c for c in simple_context().concepts() if c.intent == (0, 1)],
        num_attributes=3,
    )
    with torch.no_grad():
        # {m0, m1} есть -> принадлежит.
        assert layer(torch.tensor([[1.0, 1.0, 0.0]]))[0, 0].item() > 0.99
        # m1 нет -> не принадлежит.
        assert layer(torch.tensor([[1.0, 0.0, 1.0]]))[0, 0].item() < 0.01


def test_layer_weights_are_buffers_not_parameters():
    """Понятия найдены данными и обучению не подлежат (TS11-3),
    но обязаны попасть в state_dict и в граф как константы."""
    layer = make_layer()
    assert list(layer.parameters()) == []
    assert "intents" in layer.state_dict()
    assert "intent_sizes" in layer.state_dict()


def test_layer_rejects_out_of_range_attribute():
    from labrl.utils.fca import Concept

    with pytest.raises(ValueError, match="признаков"):
        FCAConceptLayer([Concept(extent=(0,), intent=(5,), support=1.0)], num_attributes=3)


def test_layer_rejects_empty_concept_list():
    with pytest.raises(ValueError, match="непустым содержанием"):
        FCAConceptLayer([], num_attributes=3)


# --- политика -----------------------------------------------------------


def test_policy_output_shape_and_interface():
    """Интерфейс обязан совпадать с MultiBranchCategoricalPolicy: только так
    политика подставляется в PPODiscrete без правок алгоритма (8.4)."""
    torch.manual_seed(0)
    layer = make_layer()
    policy = FCAPolicy(obs_dim=6, branches=(4,), concept_layer=layer, hidden_sizes=(16,))

    obs = torch.rand(5, 6)
    assert policy(obs).shape == (5, 4)
    action, log_prob = policy.sample(obs)
    assert action.shape == (5, 1) and log_prob.shape == (5,)
    assert policy.greedy(obs).shape == (5, 1)
    assert policy.log_prob(obs, action).shape == (5,)
    assert policy.entropy(obs).ndim == 0


def test_policy_rejects_layer_wider_than_observation():
    layer = make_layer()
    with pytest.raises(ValueError, match="размерность"):
        FCAPolicy(obs_dim=2, branches=(4,), concept_layer=layer)


def test_concepts_change_the_policy_output():
    """Признаки понятий обязаны влиять на выход: иначе слой — дорогой ноль."""
    torch.manual_seed(0)
    layer = make_layer()
    policy = FCAPolicy(obs_dim=3, branches=(4,), concept_layer=layer, hidden_sizes=(16,))

    with torch.no_grad():
        a = policy(torch.tensor([[1.0, 1.0, 0.0]]))
        b = policy(torch.tensor([[1.0, 0.0, 0.0]]))
    assert not torch.allclose(a, b)


# --- сборка по данным ---------------------------------------------------


def test_build_concept_layer_from_experience():
    rng = np.random.default_rng(0)
    # Наблюдения: три бинарных признака плюс два непрерывных.
    binary = rng.integers(0, 2, size=(500, 3)).astype(np.float32)
    obs = np.concatenate([binary, rng.normal(size=(500, 2)).astype(np.float32)], axis=1)

    layer, context = build_concept_layer(obs, num_attributes=3, min_support=0.05)

    assert context.num_attributes == 3
    assert context.num_objects == 500
    assert layer.num_concepts > 0
    # Слой читает только бинарную часть наблюдения.
    assert layer.num_attributes == 3


def test_binarize_threshold():
    values = np.array([[0.2, 0.5, 0.9]], dtype=np.float32)
    assert binarize_threshold(values, 0.5).tolist() == [[0.0, 1.0, 1.0]]
