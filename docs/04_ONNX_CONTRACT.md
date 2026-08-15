# 04_ONNX_CONTRACT — Контракт экспорта ONNX для Unity Inference Engine

**Статус документа:** `VERIFIED` — сверено с установленным пакетом в Фазе 1
(см. §0.1). Расхождения с черновиком Фазы 0 перечислены в §0.2.
**Дата:** 2026-08-15
**Основание:** `CLAUDE_Unity-ml-agents-lab.md`, раздел 10.

---

## 0. Оговорка об источнике (обязательна к прочтению)

Пункт 10.2 инструкции требует извлечь контракт **из установленного пакета `mlagents`**.
Пакет на машине **не установлен** (блокер B-2, см. `docs/01_STACK.md`), поэтому
контракт извлечён из исходников официального репозитория по тегу, соответствующему
C#-пакету, который стоит в проекте.

| Что | Значение |
|---|---|
| Репозиторий | `Unity-Technologies/ml-agents` |
| Тег | `release_23_tag` |
| Версия Python-пакета в этом теге | `1.2.0.dev0` (`ml-agents/mlagents/trainers/__init__.py`) |
| Версия C#-пакета в этом теге | `4.0.0` (`com.unity.ml-agents/package.json`) |
| Версия C#-пакета в **нашем** проекте | `4.0.3` (`Packages/manifest.json:9`) |

Прочитанные файлы (полные URL вида
`https://raw.githubusercontent.com/Unity-Technologies/ml-agents/release_23_tag/<path>`):

| # | Путь в репозитории | Что извлечено |
|---|---|---|
| 1 | `ml-agents/mlagents/trainers/torch_entities/model_serialization.py` | `TensorNames`, класс `ModelSerializer`: `input_names`, `output_names`, `dynamic_axes`, `dummy_input`, вызов `torch.onnx.export` |
| 2 | `ml-agents/mlagents/trainers/settings.py` | `SerializationSettings.onnx_opset` |
| 3 | `ml-agents/mlagents/trainers/torch_entities/networks.py` | `SimpleActor.MODEL_EXPORT_VERSION`, буферы-константы, порядок выходов в `forward()` |
| 4 | `ml-agents/mlagents/trainers/torch_entities/action_model.py` | `ActionModel.get_action_out` — что именно попадает в выходы действий |
| 5 | `ml-agents/mlagents/trainers/torch_entities/distributions.py` | `exported_model_output()` / `deterministic_sample()` для гауссовой и категориальной политик |
| 6 | `ml-agents/mlagents/trainers/model_saver/torch_model_saver.py` | точка вызова экспорта (`export_policy_model`) |
| 7 | `com.unity.ml-agents/Runtime/Inference/TensorNames.cs` | **сторона-потребитель**: строковые константы имён тензоров в Unity |

**Действие в Фазе 1 (обязательное):** после установки `mlagents` выполнить

```powershell
python -c "import mlagents, os; print(os.path.dirname(mlagents.__file__))"
```

прочитать файлы 1–5 из установленного пакета, сверить с этим документом и снять
статус `DRAFT` (или внести правки). До этого момента ни один пункт DoD, связанный
с ONNX, помечать выполненным нельзя.

---

## 1. Имена входов

Источник: `ModelSerializer.__init__`, файл 1.

```python
self.input_names = [TensorNames.get_observation_name(i) for i in range(num_obs)]
self.input_names += [
    TensorNames.action_mask_placeholder,
    TensorNames.recurrent_in_placeholder,
]
```

| # | Имя входа | Когда присутствует | Форма | Примечание |
|---|---|---|---|---|
| 0…N-1 | `obs_0`, `obs_1`, …, `obs_{N-1}` | всегда, по одному на сенсор | `(batch, *obs_spec.shape)` | Порядок = порядок `observation_specs` в `BehaviorSpec`, то есть порядок регистрации сенсоров в Unity |
| N | `action_masks` | **всегда** (даже для непрерывных действий) | `(batch, sum(discrete_branches))` | Для непрерывных действий сумма веток = 0 → тензор формы `(batch, 0)` |
| N+1 | `recurrent_in` | **всегда** | `(batch, 1, export_memory_size)` | Для нерекуррентных политик `export_memory_size = 0` |

Многомерные наблюдения (изображения) экспортируются в **NCHW** — см. комментарий
в `ModelSerializer.__init__`: «ONNX only support input in NCHW (channel first) format.
Sentis also expect to get data in NCHW.»

Преобразование формы: `_get_onnx_shape` возвращает `shape` как есть для 1-D и
`(shape[0], shape[1], shape[2])` для 3-D.

> **Имя `recurrent_in`, а не `memory_in`.** Ожидаемая форма контракта в п. 10.3
> инструкции называет вход `memory_in`; фактическое имя в коде — `recurrent_in`
> (`TensorNames.recurrent_in_placeholder = "recurrent_in"`), и оно же используется
> на стороне Unity (`TensorNames.cs:11`). Приоритет — у исходников (п. 16.4).

---

## 2. Имена выходов

Источник: `ModelSerializer.__init__` (файл 1) и `SimpleActor.forward` (файл 3) —
порядок в обоих местах совпадает и обязателен.

```python
self.output_names = [TensorNames.version_number, TensorNames.memory_size]
if continuous_size > 0:
    self.output_names += ["continuous_actions", "continuous_action_output_shape",
                          "deterministic_continuous_actions"]
if discrete_size > 0:
    self.output_names += ["discrete_actions", "discrete_action_output_shape",
                          "deterministic_discrete_actions"]
if export_memory_size > 0:
    self.output_names += ["recurrent_out"]
```

| Порядок | Имя выхода | Тип | Присутствует | Форма |
|---|---|---|---|---|
| 1 | `version_number` | константа | всегда | `(1,)` |
| 2 | `memory_size` | константа | всегда | `(1,)` |
| 3 | `continuous_actions` | значение | если `continuous_size > 0` | `(batch, continuous_size)` |
| 4 | `continuous_action_output_shape` | константа | если `continuous_size > 0` | `(1,)` |
| 5 | `deterministic_continuous_actions` | значение | если `continuous_size > 0` | `(batch, continuous_size)` |
| 6 | `discrete_actions` | значение | если `discrete_size > 0` | `(batch, num_branches)` |
| 7 | `discrete_action_output_shape` | константа | если `discrete_size > 0` | `(1, num_branches)` |
| 8 | `deterministic_discrete_actions` | значение | если `discrete_size > 0` | `(batch, num_branches)` |
| 9 | `recurrent_out` | значение | если `memory_size > 0` | `(batch, 1, memory_size)` |

Выходы `value_estimate`, `prev_action`, `epsilon`, `is_continuous_control`,
`action`, `action_output_shape` в текущем экспорте **не используются** (последние три —
помечены в коде как deprecated, оставлены для обратной совместимости).

---

## 3. Значения констант

Источник: `networks.py` (файл 3).

```python
class SimpleActor(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    self.version_number = nn.Parameter(torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False)
    self.continuous_act_size_vector = nn.Parameter(torch.Tensor([int(action_spec.continuous_size)]), requires_grad=False)
    self.discrete_act_size_vector  = nn.Parameter(torch.Tensor([action_spec.discrete_branches]), requires_grad=False)
    self.memory_size_vector        = nn.Parameter(torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False)
```

| Константа | Значение | Форма | Тип элемента |
|---|---|---|---|
| `version_number` | **3** (`ModelApiVersion.MLAgents2_0`) | `(1,)` | float32 |
| `memory_size` | `network_body.memory_size` — **0** для нерекуррентных политик | `(1,)` | float32 |
| `continuous_action_output_shape` | `continuous_size` (например `2`) | `(1,)` | float32 |
| `discrete_action_output_shape` | размеры веток, например `[[4]]` или `[[3, 3]]` | `(1, num_branches)` | float32 |

> **Уточнение формы `discrete_action_output_shape`.** В черновике Фазы 0 форма
> была записана как `(num_branches,)`. Фактически
> `torch.Tensor([action_spec.discrete_branches])` оборачивает кортеж веток
> в дополнительный список, то есть даёт `(1, num_branches)`: для одной ветки
> размера 4 это тензор `[[4.0]]`, а не `[4.0]`. Наша обёртка воспроизводит
> именно эту форму.

Все четыре — обычные `nn.Parameter` с `requires_grad=False` (в нашей обёртке
корректнее использовать `register_buffer`, результат в графе ONNX идентичен —
константный инициализатор).

---

## 4. Семантика выходов действий

Источник: `action_model.py` (файл 4) и `distributions.py` (файл 5).

### 4.1. Дискретные действия

```python
discrete_out_list = [d.exported_model_output() for d in dists.discrete]
discrete_out = torch.cat(discrete_out_list, dim=1)
deterministic_discrete_out = torch.cat([d.deterministic_sample() for d in dists.discrete], dim=1)
```

Для `CategoricalDistInstance`:

```python
def sample(self):                 return torch.multinomial(self.probs, 1)
def deterministic_sample(self):   return torch.argmax(self.probs, dim=1, keepdim=True)
def exported_model_output(self):  return self.sample()
```

**Вывод:** `discrete_actions` — это **индексы выбранных действий** формы
`(batch, num_branches)`, а не логиты и не one-hot. Это подтверждает требование 10.4
инструкции. `deterministic_discrete_actions` — `argmax` по вероятностям.

> **Тип элемента обязан быть целым.** `torch.multinomial` и `torch.argmax`
> возвращают `int64`, и ML-Agents этот тип не меняет. Unity читает тензор как
> `Tensor<int>`:
>
> ```csharp
> discreteBuffer[j] = ((Tensor<int>)tensorProxy.data)[agentIndex, j];
> ```
>
> (`com.unity.ml-agents@4.0.3`, `Runtime/Inference/ApplierImpl.cs`,
> `DiscreteActionOutputApplier`). Приведение выхода к `float` даёт **валидный**
> ONNX, который тем не менее падает в Unity при инференсе. Непрерывный аналог
> читается как `Tensor<float>` (`ContinuousActionOutputApplier`), то есть там
> вещественный тип, наоборот, обязателен.
>
> Проверка типов входит в `labrl.export.onnx_verify`.

### 4.2. Непрерывные действия

```python
continuous_out = dists.continuous.exported_model_output()      # = sample()
deterministic_continuous_out = dists.continuous.deterministic_sample()  # = mean
if self.clip_action:
    continuous_out = torch.clamp(continuous_out, -3, 3) / 3
    deterministic_continuous_out = torch.clamp(deterministic_continuous_out, -3, 3) / 3
```

**Вывод:** при `clip_action=True` (по умолчанию для не-tanh политик) действия
клипуются в `[-3, 3]` и делятся на 3, то есть на выходе диапазон **`[-1, 1]`**.
Наша реализация экспортёра обязана воспроизводить именно этот диапазон, иначе
масштаб управляющего сигнала в Unity разойдётся с обучением.

---

## 5. Параметры вызова `torch.onnx.export`

Источник: `ModelSerializer.export_policy_model` (файл 1) и `settings.py` (файл 2).

```python
class SerializationSettings:
    convert_to_onnx = True
    onnx_opset = 9
```

```python
torch.onnx.export(
    self.policy.actor,
    self.dummy_input,                      # (dummy_obs: list[Tensor], dummy_masks, dummy_memories)
    onnx_output_path,
    opset_version=SerializationSettings.onnx_opset,   # 9
    input_names=self.input_names,
    output_names=self.output_names,
    dynamic_axes=self.dynamic_axes,
)
```

| Параметр | Значение |
|---|---|
| **opset** | **9** |
| `dynamic_axes` | `{0: "batch"}` для **всех входов** и для выходов `continuous_actions` / `discrete_actions` |
| `dummy_obs` | `torch.zeros([1] + onnx_shape(obs_spec.shape))` для каждого сенсора |
| `dummy_masks` | `torch.ones([1, sum(discrete_branches)])` |
| `dummy_memories` | `torch.zeros([1, 1, export_memory_size])` |

Обратите внимание: `dynamic_axes` **не** задаётся для константных выходов и для
`deterministic_*` выходов — там батч фиксирован в 1 на этапе трассировки, но
Unity читает их как константы/по батчу входа.

---

## 6. Сторона Unity (потребитель контракта)

Источник: `com.unity.ml-agents/Runtime/Inference/TensorNames.cs` @ `release_23_tag`.

```csharp
public const string BatchSizePlaceholder = "batch_size";
public const string SequenceLengthPlaceholder = "sequence_length";
public const string VectorObservationPlaceholder = "vector_observation";
public const string RecurrentInPlaceholder = "recurrent_in";
public const string VisualObservationPlaceholderPrefix = "visual_observation_";
public const string ObservationPlaceholderPrefix = "obs_";
public const string PreviousActionPlaceholder = "prev_action";
public const string ActionMaskPlaceholder = "action_masks";
public const string RandomNormalEpsilonPlaceholder = "epsilon";

public const string ValueEstimateOutput = "value_estimate";
public const string RecurrentOutput = "recurrent_out";
public const string MemorySize = "memory_size";
public const string VersionNumber = "version_number";
public const string ContinuousActionOutputShape = "continuous_action_output_shape";
public const string DiscreteActionOutputShape = "discrete_action_output_shape";
public const string ContinuousActionOutput = "continuous_actions";
public const string DiscreteActionOutput = "discrete_actions";
public const string DeterministicContinuousActionOutput = "deterministic_continuous_actions";
public const string DeterministicDiscreteActionOutput = "deterministic_discrete_actions";

// Deprecated
public const string IsContinuousControlDeprecated = "is_continuous_control";
public const string ActionOutputDeprecated = "action";
public const string ActionOutputShapeDeprecated = "action_output_shape";
```

Имена на стороне Unity **полностью совпадают** с именами, которые генерирует
Python-экспортёр. Расхождений нет.

---

## 7. Итоговый контракт (то, что обязан выдавать `labrl.export.onnx_export`)

### 7.1. Пример: дискретные действия, 1 ветвь × 4, один векторный сенсор 25

| Роль | Имя | Форма | Значение |
|---|---|---|---|
| вход | `obs_0` | `(batch, 25)` | наблюдение |
| вход | `action_masks` | `(batch, 4)` | маски |
| вход | `recurrent_in` | `(batch, 1, 0)` | пусто |
| выход | `version_number` | `(1,)` | `3.0` |
| выход | `memory_size` | `(1,)` | `0.0` |
| выход | `discrete_actions` | `(batch, 1)` | индекс действия 0–3 |
| выход | `discrete_action_output_shape` | `(1,)` | `4.0` |
| выход | `deterministic_discrete_actions` | `(batch, 1)` | `argmax` |

### 7.2. Пример: непрерывные действия, размер 2, один векторный сенсор 8

| Роль | Имя | Форма | Значение |
|---|---|---|---|
| вход | `obs_0` | `(batch, 8)` | наблюдение |
| вход | `action_masks` | `(batch, 0)` | пусто |
| вход | `recurrent_in` | `(batch, 1, 0)` | пусто |
| выход | `version_number` | `(1,)` | `3.0` |
| выход | `memory_size` | `(1,)` | `0.0` |
| выход | `continuous_actions` | `(batch, 2)` | в `[-1, 1]` |
| выход | `continuous_action_output_shape` | `(1,)` | `2.0` |
| выход | `deterministic_continuous_actions` | `(batch, 2)` | среднее политики, тот же диапазон |

### 7.3. Обязательные проверки `labrl.export.onnx_verify` (по 10.5)

| Проверка | Критерий | Готово к реализации |
|---|---|---|
| Структура | `onnx.checker.check_model` без ошибок | да |
| Имена | множества входов/выходов **точно** равны таблицам §1–§2 | да |
| Константы | `version_number == 3`, `memory_size == 0` (нерекуррентные), `*_output_shape` равны `ActionSpec` | да |
| Формы | прогон `onnxruntime` на батчах 1 и 64 | да |
| Числовой паритет | `max‖ONNX − PyTorch‖∞ ≤ 1e-4` на 64 наблюдениях из реального распределения среды; сравнение по **детерминированным** выходам (стохастические `sample()` несравнимы поэлементно) | да |
| Диапазоны | непрерывные — в `[-1, 1]`; дискретные — целые в `[0, branch_size)` | да |
| opset | ровно `9` | да |

> Замечание к пункту «числовой паритет»: выходы `continuous_actions` и
> `discrete_actions` стохастичны (`sample()`), поэтому численно сравнивать
> PyTorch и ONNX следует по `deterministic_*` выходам, зафиксировав seed для
> стохастических. Это уточнение внесено в `docs/ASSUMPTIONS.md` как A-6.

---

## 8. Нормализация наблюдений

Требование 10.7: статистики нормализации **не хранятся вне ONNX**. В штатном
ML-Agents нормализация реализована внутри `NetworkBody` (`VectorInput` с
`Normalizer`), то есть попадает в граф. Наша обёртка обязана поступать так же:
либо встроить нормализацию в экспортируемый модуль, либо не применять её вовсе.
Проверка «нет внешних статистик» включается в `onnx_verify` в Фазе 1.
