"""Экспорт обученной политики в ONNX, пригодный для Unity Inference Engine.

Критический модуль проекта (раздел 10 инструкции). «Обычный»
``torch.onnx.export`` даёт граф, который Unity **не примет**: ML-Agents требует
строго определённых имён входов и выходов и служебных выходов-констант.

Контракт зафиксирован в ``docs/04_ONNX_CONTRACT.md`` и извлечён из исходников
пакета ``mlagents`` — здесь он воспроизводится, а не изобретается.

Ключевая структура::

    входы:  obs_0 … obs_{N-1}
            [action_masks]  — только при дискретных действиях
            [recurrent_in]  — только при memory_size > 0
    выходы: version_number, memory_size,
            [continuous_actions, continuous_action_output_shape, deterministic_continuous_actions]
            [discrete_actions,   discrete_action_output_shape,   deterministic_discrete_actions]
            [recurrent_out]

Квадратные скобки здесь не «опционально по вкусу», а фактическое поведение:
``torch.onnx.export`` выбрасывает из графа неиспользуемые входы, и Unity ждёт
ровно такой граф (см. :func:`contract_input_names`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import torch
import torch.nn as nn

#: Версия API модели ML-Agents (`SimpleActor.MODEL_EXPORT_VERSION`,
#: соответствует `ModelApiVersion.MLAgents2_0`). Проверяется при верификации.
MODEL_EXPORT_VERSION = 3

#: Версия opset, с которой ML-Agents экспортирует модели
#: (`SerializationSettings.onnx_opset`). Выбирать произвольно запрещено (10.3).
ONNX_OPSET = 9

#: Диапазон, в который ML-Agents приводит непрерывные действия перед выдачей
#: в Unity: clamp(-3, 3) / 3 -> [-1, 1] (контракт §4.2).
CONTINUOUS_CLIP = 3.0


class _HasActionSpec(Protocol):
    """Утиный тип: и ``mlagents_envs.base_env.ActionSpec``, и :class:`ActionSpecLite`."""

    continuous_size: int
    discrete_branches: tuple[int, ...]


@dataclass(frozen=True)
class ActionSpecLite:
    """Описание пространства действий без зависимости от живой среды.

    Нужен, чтобы экспорт и его тесты работали без запущенного Unity.
    """

    continuous_size: int = 0
    discrete_branches: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.continuous_size < 0:
            raise ValueError("continuous_size не может быть отрицательным")
        if any(b <= 0 for b in self.discrete_branches):
            raise ValueError(f"размеры дискретных веток должны быть > 0: {self.discrete_branches}")
        if self.continuous_size == 0 and not self.discrete_branches:
            raise ValueError("пространство действий пусто: нужен continuous_size > 0 или хотя бы одна ветка")


def contract_input_names(spec: _HasActionSpec, num_obs: int, memory_size: int = 0) -> list[str]:
    """Имена входов **фактического графа** в порядке контракта (§1).

    Тонкость, которую легко принять за ошибку. ``ModelSerializer`` из ML-Agents
    объявляет ``input_names`` всегда как ``obs_i + action_masks + recurrent_in``,
    но ``torch.onnx.export`` **выбрасывает из графа входы, которые модель
    не использует**: маски нужны только дискретной политике, память — только
    рекуррентной. Поэтому в экспортированном ML-Agents графе непрерывной
    нерекуррентной политики есть только ``obs_i``.

    Unity ждёт ровно этого: ``action_masks`` требуется, лишь когда у модели есть
    дискретные выходы, а ``recurrent_in`` — лишь когда ``memory_size > 0``
    (``com.unity.ml-agents@4.0.3``, ``SentisModelParamLoader.CheckInputTensorPresence``).
    """
    names = [f"obs_{i}" for i in range(num_obs)]
    if len(spec.discrete_branches) > 0:
        names.append("action_masks")
    if memory_size > 0:
        names.append("recurrent_in")
    return names


def contract_output_names(spec: _HasActionSpec, memory_size: int = 0) -> list[str]:
    """Имена выходов в порядке контракта (§2). Порядок обязателен."""
    names = ["version_number", "memory_size"]
    if spec.continuous_size > 0:
        names += ["continuous_actions", "continuous_action_output_shape",
                  "deterministic_continuous_actions"]
    if len(spec.discrete_branches) > 0:
        names += ["discrete_actions", "discrete_action_output_shape",
                  "deterministic_discrete_actions"]
    if memory_size > 0:
        names += ["recurrent_out"]
    return names


def _concat_obs(obs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Сборка нескольких векторных сенсоров в один вход политики.

    Порядок конкатенации — порядок сенсоров в Unity, он же порядок ``obs_i``.
    Визуальные наблюдения (4-D тензоры) этой функцией не поддерживаются:
    для них политика должна принимать список тензоров, что появится вместе
    с первой средой с камерой.
    """
    if len(obs) == 1:
        return obs[0]
    if any(t.dim() != 2 for t in obs):
        raise NotImplementedError(
            "конкатенация по умолчанию рассчитана на векторные сенсоры (B, d); "
            "для визуальных наблюдений передайте свой obs_combiner"
        )
    return torch.cat(list(obs), dim=1)


class MLAgentsPolicyWrapper(nn.Module):
    """Обёртка политики, выдающая полный набор выходов контракта.

    Args:
        policy: обученный ``nn.Module``. Для дискретных действий должен
            возвращать тензор ``(B, sum(discrete_branches))`` — Q-значения
            (``strategy="greedy"``) или логиты (``strategy="categorical"``);
            для непрерывных — ``(B, continuous_size)``, среднее политики
            в «сырых» единицах, до клиппинга.
        action_spec: описание пространства действий среды.
        memory_size: размер рекуррентной памяти. Ненулевые значения пока
            не поддерживаются (рекуррентных политик в курсе нет).
        strategy: как получить ``*_actions`` для дискретного случая.
            ``greedy`` — argmax (политика DQN и производных: стохастика живёт
            в ε-жадности на стороне Python, а не в графе);
            ``categorical`` — сэмплирование из softmax (policy-gradient методы).
        obs_combiner: как склеить наблюдения нескольких сенсоров в вход политики.

    Замечание про нормализацию (требование 10.7): если политика нормализует
    наблюдения, нормализация обязана быть частью ``policy`` — то есть попасть
    в граф. Хранить статистики снаружи ONNX запрещено.
    """

    def __init__(
        self,
        policy: nn.Module,
        action_spec: _HasActionSpec,
        memory_size: int = 0,
        strategy: str = "greedy",
        obs_combiner: Callable[[Sequence[torch.Tensor]], torch.Tensor] = _concat_obs,
    ) -> None:
        super().__init__()
        if memory_size != 0:
            raise NotImplementedError(
                "рекуррентные политики (memory_size > 0) не поддерживаются: "
                "выход recurrent_out потребует отдельной реализации"
            )
        if strategy not in ("greedy", "categorical"):
            raise ValueError(f"strategy должна быть greedy|categorical, получено {strategy!r}")
        if action_spec.continuous_size > 0 and len(action_spec.discrete_branches) > 0:
            raise NotImplementedError(
                "гибридное пространство действий (continuous + discrete) не поддерживается"
            )

        self.policy = policy
        self.continuous_size = int(action_spec.continuous_size)
        self.discrete_branches = tuple(int(b) for b in action_spec.discrete_branches)
        self.memory_size = int(memory_size)
        self.strategy = strategy
        self.obs_combiner = obs_combiner

        # Выходы-константы. В ML-Agents это nn.Parameter(requires_grad=False);
        # register_buffer даёт в графе ONNX тот же константный инициализатор,
        # но не попадает в список обучаемых параметров.
        self.register_buffer("version_number", torch.tensor([MODEL_EXPORT_VERSION], dtype=torch.float32))
        self.register_buffer("memory_size_vector", torch.tensor([self.memory_size], dtype=torch.float32))
        self.register_buffer(
            "continuous_act_size_vector",
            torch.tensor([self.continuous_size], dtype=torch.float32),
        )
        self.register_buffer(
            "discrete_act_size_vector",
            torch.tensor([list(self.discrete_branches)], dtype=torch.float32),
        )

    # --- вспомогательное -------------------------------------------------

    @property
    def num_branches(self) -> int:
        return len(self.discrete_branches)

    @property
    def mask_size(self) -> int:
        """Ширина входа ``action_masks``: сумма веток (0 для непрерывных)."""
        return int(sum(self.discrete_branches))

    def _split_branches(self, logits: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(logits, list(self.discrete_branches), dim=1))

    def _apply_mask(self, branch_logits: torch.Tensor, branch_mask: torch.Tensor) -> torch.Tensor:
        """Запрещённые действия получают −1e8 вместо −inf.

        −inf даёт NaN в softmax, если запрещены все действия ветки; −1e8
        сохраняет корректный argmax и не ломает численную стабильность.
        """
        return branch_logits + (branch_mask - 1.0) * 1e8

    # --- прямой проход ---------------------------------------------------

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Порядок аргументов и результатов — строго по контракту.

        Args:
            *inputs: ``obs_0 … obs_{N-1}``, затем ``action_masks``
                ``(B, mask_size)``, затем ``recurrent_in`` ``(B, 1, memory_size)``.

        Returns:
            Кортеж выходов в порядке :func:`contract_output_names`.
        """
        if len(inputs) < 3:
            raise ValueError(
                f"ожидались минимум 3 входа (obs_0, action_masks, recurrent_in), получено {len(inputs)}"
            )
        obs = inputs[:-2]
        action_masks = inputs[-2]

        net_out = self.policy(self.obs_combiner(obs))
        outputs: list[torch.Tensor] = [self.version_number, self.memory_size_vector]

        if self.continuous_size > 0:
            clipped = torch.clamp(net_out, -CONTINUOUS_CLIP, CONTINUOUS_CLIP) / CONTINUOUS_CLIP
            # Детерминированный выход — среднее политики; стохастический для
            # непрерывного случая формируется алгоритмом (шум добавляется в
            # Python при сборе опыта), поэтому в графе они совпадают.
            outputs += [clipped, self.continuous_act_size_vector, clipped]

        if self.num_branches > 0:
            masked = [
                self._apply_mask(bl, bm)
                for bl, bm in zip(
                    self._split_branches(net_out),
                    torch.split(action_masks, list(self.discrete_branches), dim=1),
                )
            ]
            # Тип выхода обязан остаться целым. Unity читает этот тензор как
            # Tensor<int> (com.unity.ml-agents@4.0.3,
            # Runtime/Inference/ApplierImpl.cs, DiscreteActionOutputApplier),
            # поэтому приведение к float ломает инференс на стороне Unity,
            # хотя сам ONNX при этом остаётся валидным.
            deterministic = torch.cat([torch.argmax(m, dim=1, keepdim=True) for m in masked], dim=1)
            if self.strategy == "greedy":
                sampled = deterministic
            else:
                sampled = torch.cat(
                    [torch.multinomial(torch.softmax(m, dim=1), 1) for m in masked], dim=1
                )
            outputs += [sampled, self.discrete_act_size_vector, deterministic]

        return tuple(outputs)


def export_policy_to_onnx(
    policy: nn.Module,
    action_spec: _HasActionSpec,
    obs_shapes: Sequence[tuple[int, ...]],
    output_path: str | Path,
    memory_size: int = 0,
    strategy: str = "greedy",
) -> Path:
    """Экспортирует политику в ONNX по контракту ML-Agents.

    Args:
        policy: обученный модуль политики (``Algo.policy_module()``).
        action_spec: пространство действий среды.
        obs_shapes: формы наблюдений по сенсорам, **без** батча, в порядке
            регистрации сенсоров в Unity.
        output_path: куда писать ``.onnx``.
        memory_size: размер рекуррентной памяти (сейчас поддерживается только 0).
        strategy: ``greedy`` для DQN-подобных, ``categorical`` для
            policy-gradient методов.

    Returns:
        Путь к записанному файлу.

    Экспорт **не** проверяет модель — это делает
    :func:`labrl.export.onnx_verify.verify_onnx_model`, и его провал блокирует
    приёмку (требование 10.5).
    """
    # Экспорт всегда выполняется на CPU. Причины две: трассировка модуля на GPU
    # требует, чтобы фиктивные входы тоже были на GPU (иначе RuntimeError про
    # разные устройства), а результат обязан быть одинаковым независимо от того,
    # была ли на машине видеокарта. Исходное устройство восстанавливается —
    # обучение после экспорта продолжается там же, где шло.
    original_device = _module_device(policy)
    policy.to("cpu")
    wrapper = MLAgentsPolicyWrapper(policy, action_spec, memory_size, strategy).eval()

    dummy_obs = tuple(torch.zeros((1, *shape), dtype=torch.float32) for shape in obs_shapes)
    dummy_masks = torch.ones((1, max(wrapper.mask_size, 0)), dtype=torch.float32)
    dummy_memory = torch.zeros((1, 1, memory_size), dtype=torch.float32)

    input_names = contract_input_names(action_spec, len(obs_shapes), memory_size)
    output_names = contract_output_names(action_spec, memory_size)

    # dynamic_axes по батчу — для всех входов и для стохастических выходов
    # действий: именно так делает ModelSerializer (контракт §5).
    dynamic_axes: dict[str, dict[int, str]] = {name: {0: "batch"} for name in input_names}
    for name in ("continuous_actions", "discrete_actions"):
        if name in output_names:
            dynamic_axes[name] = {0: "batch"}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (*dummy_obs, dummy_masks, dummy_memory),
                str(out),
                opset_version=ONNX_OPSET,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    finally:
        policy.to(original_device)
    return out


def _module_device(module: nn.Module) -> torch.device:
    """Устройство модуля. У модуля без параметров — CPU."""
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")
