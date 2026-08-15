# 07_TROUBLESHOOTING — Ошибки и решения

Документ накапливается по ходу работы. Сюда попадают **только фактически
встреченные** проблемы с их фактическим решением, а не гипотезы.

Формат записи: симптом → причина → решение → откуда известно.

---

## Установка окружения

### T-1. `mlagents-envs` не ставится под Python 3.11+

**Симптом.** `pip install mlagents-envs` сообщает, что ни одна версия не подходит.

**Причина.** `setup.py` пакета содержит `python_requires=">=3.10.1,<=3.10.12"`.
Источник: `ml-agents-envs/setup.py` тега `release_23_tag`.

**Решение.** Установить Python 3.10.11 и создать окружение именно на нём:

```powershell
winget install --id Python.Python.3.10 --version 3.10.11 --scope user --silent
py -3.10 -m venv python\.venv
```

**Дата:** 2026-08-15.

### T-2. Клонирование `ml-agents` с GitHub идёт со скоростью ~16 КБ/с

**Симптом.** `pip install "git+https://github.com/Unity-Technologies/ml-agents.git@release_23_tag#subdirectory=ml-agents-envs"`
висит на стадии `Cloning …` десятки минут; каталог клона растёт примерно
на 1 МБ в минуту. `git clone --depth 1` ведёт себя так же.

**Причина.** Ограничение скорости именно на git/codeload-эндпоинтах GitHub.
Проверено: `raw.githubusercontent.com` и `packages.unity.com` в тот же момент
отдают данные на нормальной скорости (32 МБ пакета Unity скачались за секунды).

**Решение.** Скачать файлы подкаталога `ml-agents-envs` пофайлово через
`raw.githubusercontent.com` (список файлов — через `api.github.com`
`git/trees/release_23_tag:ml-agents-envs?recursive=1`) и поставить из локального
каталога. Код при этом ровно тот же — тег определяет содержимое файлов.

Проверка, что установлен нужный код:

```powershell
.\python\.venv\Scripts\python.exe -c "import mlagents_envs; print(mlagents_envs.__version__, mlagents_envs.__release_tag__)"
```

`__release_tag__` обязан быть `release_23_tag`.

**Дата:** 2026-08-15.

### T-3. Конфликт версий `numpy` и `protobuf` в стеке

**Симптом.** Свежие `onnx`, `onnxruntime`, `tensorboard` тянут `protobuf>=4`
и `numpy>=2`, а `mlagents-envs` требует `numpy>=1.23.5,<1.24.0` и
`protobuf>=3.6,<3.21`.

**Причина.** `mlagents-envs@release_23_tag` заморожен на старых границах;
источник — его `setup.py`.

**Решение.** Границы всего стека подчинены ML-Agents; они зафиксированы
в `python/pyproject.toml` с объяснением. Проверка — `pip check`, вывод
приложен к `docs/01_STACK.md` (требование 4.4).

**Дата:** 2026-08-15.

---

## Unity

### T-4. `com.unity.ml-agents` 4.0.0 не компилируется на Unity 6000.5

**Симптом.** Ошибка компиляции в `Match3ActuatorComponent.cs`:
`Object.GetInstanceID()` — obsolete-as-error.

**Решение.** Использовать 4.0.3+. Зафиксировано в `Packages/manifest.json`.

### T-5. `TensorNames` из ML-Agents недоступен из кода проекта

**Симптом.** `Unity.MLAgents.Inference.TensorNames` не виден из скриптов
в `Assets/`.

**Причина.** Класс объявлен `internal` (проверено в
`com.unity.ml-agents@4.0.3/Runtime/Inference/TensorNames.cs`). То же относится
к `SentisModelInfo`, поэтому и `SentisModelParamLoader.CheckModel` фактически
недоступен снаружи сборки пакета.

**Решение.** `Assets/Shared/Editor/OnnxContractCheck.cs` дублирует имена
тензоров строковыми константами, сверенными с исходником пакета, и работает
через публичный `Unity.InferenceEngine.ModelLoader` / `Model.inputs` / `Model.outputs`.

**Дата:** 2026-08-15.

---

## ONNX и инференс в Unity

*(Раздел пополняется в Фазе 2, когда появится первая обученная модель.
Типовые причины расхождения Python и Unity, которые предстоит проверять
по требованию 10.6: рассогласование порядка наблюдений, отсутствующая
нормализация, различие в частоте принятия решений.)*

---

## Обучение

*(Пополняется в Фазе 2.)*
