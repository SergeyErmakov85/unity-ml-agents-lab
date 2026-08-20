# 01_STACK — Верификация технологического стека

**Дата:** 2026-08-15; раздел 7 добавлен 2026-08-20
**Основание:** `CLAUDE_Unity-ml-agents-lab.md`, раздел 4.
**Правило:** ни одна версия не взята из памяти модели. Для каждой строки указан
источник — файл со строкой либо команда с выводом.

Статусы: `VERIFIED` — подтверждено выводом команды/чтением файла;
`NOT VERIFIED` — источник прочитан, но фактическая работоспособность не проверена;
`BLOCKED` — проверка невозможна до устранения блокера.

---

## 1. Unity

### 1.1. Версия проекта

Источник: `ProjectSettings/ProjectVersion.txt`, строки 1–2.

```
m_EditorVersion: 6000.5.4f1
m_EditorVersionWithRevision: 6000.5.4f1 (d550df8bd089)
```

Статус: `VERIFIED` (файл прочитан).

### 1.2. Установленные редакторы

Команда:

```powershell
Get-ChildItem 'C:\Program Files\Unity\Hub\Editor' -Directory |
  ForEach-Object { $exe = Join-Path $_.FullName 'Editor\Unity.exe'; '{0,-16} Unity.exe={1}' -f $_.Name, (Test-Path $exe) }
```

Вывод:

```
2021.3.45f1      Unity.exe=True
2023.2.13f1      Unity.exe=True
2023.2.20f1      Unity.exe=True
6000.0.35f1      Unity.exe=True
6000.3.10f1      Unity.exe=True
6000.5.0a8       Unity.exe=True
```

Дополнительная проверка альтернативных путей установки:

```powershell
$cfg="$env:APPDATA\UnityHub\secondaryInstallPath.json"   # содержимое: ""
$ed ="$env:APPDATA\UnityHub\editors-v2.json"             # файл отсутствует
```

**Вывод:** редактор `6000.5.4f1`, требуемый проектом, **на машине отсутствует**.
Ближайшие установленные — `6000.5.0a8` (альфа той же линии) и `6000.3.10f1`.
Открытие проекта любым другим редактором приведёт к апгрейду проекта.

Статус: **BLOCKED (B-1)**.

Примечание: `CLAUDE.md` репозитория указывает команду сборки с путём
`C:\Program Files\Unity\Hub\Editor\6000.5.4f1\Editor\Unity.exe` — этот путь
на данный момент не существует.

---

## 2. Пакеты Unity

Источник: `Packages/manifest.json` (строки указаны).

| Пакет | Версия | Строка |
|---|---|---|
| `com.unity.ml-agents` | 4.0.3 | 9 |
| `com.unity.ai.inference` | 2.6.1 | 4 |
| `com.unity.render-pipelines.universal` | 17.5.0 | 12 |
| `com.unity.inputsystem` | 1.19.0 | 10 |
| `com.unity.ai.navigation` | 2.0.13 | 5 |
| `com.unity.test-framework` | 1.7.0 | 13 |

`Packages/packages-lock.json` присутствует (14 979 Б).

**Barracuda не используется** — требование 4.3 соблюдено: инференс-движок это
`com.unity.ai.inference` 2.6.1 (наследник Sentis).

Разрешение пакетов не проверено: каталог `Library/` отсутствует
(`Test-Path 'C:\unity-ml-agents-lab\Library'` → `False`), глобальный кэш
`%LOCALAPPDATA%\Unity\cache\packages\packages.unity.com` не содержит
`com.unity.ml-agents` (содержит `com.unity.sentis@2.1.0` от других проектов).

Статус: `NOT VERIFIED` (следствие B-1).

---

## 3. Python

### 3.1. Интерпретаторы на машине

```powershell
(Get-Command python -All).Source
```
```
C:\Users\<user>\AppData\Local\Programs\Python\Python311\python.exe
C:\Users\<user>\AppData\Local\Microsoft\WindowsApps\python.exe
```

```powershell
python --version
```
```
Python 3.11.4
```

```powershell
py -0p
```
```
 -V:3.14 *        C:\Python314\python.exe
 -V:3.11          C:\Users\<user>\AppData\Local\Programs\Python\Python311\python.exe
```

Проверка 3.14: `py -3.14 --version` → `Unable to create process using
'C:\Python314\python.exe --version'` (интерпретатор зарегистрирован, но неработоспособен).

```powershell
conda env list
```
```
no conda
```

**Вывод:** доступен ровно один рабочий интерпретатор — **Python 3.11.4**.
Anaconda/conda, которую предписывает урок 1.2 курса, не установлена.

### 3.2. Установленные пакеты (Python 3.11.4)

```powershell
python -m pip list
```
```
Package         Version
--------------- -------
et-xmlfile      1.1.0
numpy           1.25.1
openpyxl        3.1.2
pandas          2.0.3
pip             23.1.2
python-dateutil 2.8.2
pytz            2023.3
setuptools      65.5.0
six             1.16.0
tzdata          2023.3
```

**Отсутствуют полностью:** `mlagents`, `mlagents-envs`, `torch`, `onnx`,
`onnxruntime`, `tensorboard`, `protobuf`, `grpcio`.

Статус: **BLOCKED (B-2)**.

### 3.3. `pip check`

Требование 4.4 — выполнить `pip check` и приложить вывод.
На текущий момент проверять нечего: целевое окружение `python/.venv` ещё не создано
(его создание относится к Фазе 1, а установка пакетов запрещена пунктом 0.3 до
завершения Фазы 0). `pip check` будет выполнен и приложен сюда в Фазе 1.

Статус: **BLOCKED (B-2)**.

---

## 4. Требования пакета `mlagents` (источник — setup.py официального репозитория)

Так как пакет не установлен, требования прочитаны из исходников
`Unity-Technologies/ml-agents` по тегу `release_23_tag`
(`https://raw.githubusercontent.com/Unity-Technologies/ml-agents/release_23_tag/ml-agents/setup.py`).

```python
python_requires=">=3.10.1,<=3.10.12"
install_requires=[
    "grpcio>=1.11.0,<=1.53.2",
    "h5py>=2.9.0",
    f"mlagents_envs=={VERSION}",
    "numpy>=1.23.5,<1.24.0",
    "Pillow>=4.2.1",
    "protobuf>=3.6,<3.21",
    "pyyaml>=3.1.0",
    "torch>=2.1.1",
    "tensorboard>=2.14",
    "six>=1.16",
    "cattrs>=1.1.0,<1.7; python_version>='3.8'",
    "attrs>=19.3.0",
    "huggingface_hub>=0.14",
    'pypiwin32==223;platform_system=="Windows"',
    "onnx==1.15.0",
]
```

`ml-agents-envs/setup.py` — те же ограничения `python_requires` и `numpy`, плюс
`gym>=0.21.0`, `pettingzoo==1.15.0`, `cloudpickle`, `filelock>=3.4.0`.

Версия из `ml-agents/mlagents/trainers/__init__.py` @ `release_23_tag`:
`__version__ = "1.2.0.dev0"`.

### 4.1. Три жёстких следствия

1. **Python 3.11.4 непригоден** для `mlagents`/`mlagents_envs`. Требуется 3.10.1–3.10.12.
2. **Установленный `numpy` 1.25.1 несовместим** (нужен `>=1.23.5,<1.24.0`). В отдельном
   `python/.venv` конфликта не возникнет.
3. **PyPI не содержит нужную версию.** Проверено:

   ```powershell
   python -m pip index versions mlagents
   ```
   ```
   mlagents (0.28.0)
   Available versions: 0.28.0, 0.27.0, 0.26.0, ... 0.4.0
   ```
   ```powershell
   python -m pip index versions mlagents-envs
   ```
   ```
   mlagents-envs (0.28.0)
   Available versions: 0.28.0, 0.27.0, ... 0.10.0
   ```

   Последняя опубликованная на PyPI версия — `0.28.0` (ML-Agents Release 19).
   Версии 1.x (Release 20–23) на PyPI отсутствуют и ставятся только из git —
   ровно как учит урок 1.2 курса («ML-Agents Release 22 через git»).

### 4.2. Соответствие C#-пакета и Python-пакета

Проверено по тегам репозитория (`https://api.github.com/repos/Unity-Technologies/ml-agents/tags`):
существуют `release_23_tag`, `release_22`, `release_21`, `release_20`, `release_19`, …

`com.unity.ml-agents/package.json` @ `release_23_tag`:

```json
{ "name": "com.unity.ml-agents", "version": "4.0.0", "unity": "6000.0",
  "dependencies": { "com.unity.ai.inference": "2.2.1", ... } }
```

В проекте установлен **4.0.3** — патч-версия, опубликованная в реестр Unity позже
тега. Соответствующий Python-пакет — ветка `release_23_tag` (`1.2.0.dev0`) либо более
поздний `develop`. Точное соответствие «C# 4.0.3 ↔ Python-версия» подлежит проверке
в Фазе 1 после установки (совместимость проверяется по номеру communication API при
первом подключении `mlagents_envs` к билду).

Статус: `NOT VERIFIED`.

---

## 5. Итоговая таблица стека

| Компонент | Ожидалось (4.3 инструкции) | Фактически | Статус |
|---|---|---|---|
| Unity | 2023.2 LTS или новее | 6000.5.4f1 в проекте, **не установлен** | BLOCKED B-1 |
| `com.unity.ml-agents` | 3.x (Release 22) | **4.0.3** (новее ожидаемого) | NOT VERIFIED |
| Инференс-движок | Inference Engine / Sentis, не Barracuda | `com.unity.ai.inference` 2.6.1 ✔ | NOT VERIFIED |
| URP | требуется | 17.5.0 ✔ | NOT VERIFIED |
| Python | 3.10.x | 3.11.4 (единственный рабочий) | BLOCKED B-2 |
| `mlagents-envs` | 1.1.x | не установлен; на PyPI максимум 0.28.0 | BLOCKED B-2 |
| PyTorch | 2.2.x | не установлен | BLOCKED B-2 |
| `onnx` / `onnxruntime` / `tensorboard` | требуются | не установлены | BLOCKED B-2 |
| `numpy` | совместимая с `mlagents-envs` | 1.25.1 (вне диапазона `<1.24.0`) | конфликт |
| Git LFS | требуется по `.gitattributes` | `git-lfs/3.7.0` ✔ | VERIFIED |

**Расхождение с ожиданиями инструкции 4.3:** установленный C#-пакет — 4.0.x, а не 3.x;
это делает актуальным Python-стек ветки `release_23_tag`, а не Release 22.
Решение требуется от пользователя (вопрос Q1 в `PLAN.md`).

---

## 6. Фаза 1: стек фактически установлен и проверен

**Дата:** 2026-08-15. Раздел заменяет статусы `BLOCKED`/`NOT VERIFIED` выше
для всего, что относится к Python. Блокеры B-1 и B-2 сняты.

### 6.1. Что и как установлено

| Компонент | Версия | Как установлено |
|---|---|---|
| Python | 3.10.11 | `winget install --id Python.Python.3.10 --version 3.10.11 --scope user` |
| Окружение | `python/.venv` | `py -3.10 -m venv python\.venv` |
| PyTorch | 2.2.1+cu121 | `pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121` |
| `mlagents_envs` | 1.2.0.dev0 (`release_23_tag`) | из локальной копии файлов тега, см. A-14 и T-2 |
| `mlagents` | 1.2.0.dev0 (`release_23_tag`) | там же |
| `labrl` | 0.1.0 | `pip install -e python --no-deps` |

Про способ установки ML-Agents. Штатная команда
`pip install "git+https://github.com/Unity-Technologies/ml-agents.git@release_23_tag#subdirectory=…"`
на этой машине упирается в ограничение скорости git-эндпоинтов GitHub (~16 КБ/с,
измерено; `raw.githubusercontent.com` и `packages.unity.com` при этом работают
на полной скорости). Файлы обоих подкаталогов тега скачаны с `raw` и установлены
из локального каталога. Содержимое определяется тегом, поэтому код идентичен;
подробности — `docs/07_TROUBLESHOOTING.md`, T-2.

### 6.2. Вывод команд верификации

```
> python -c "import sys,numpy,torch,onnx,onnxruntime,mlagents,mlagents_envs,google.protobuf as pb,tensorboard; ..."
python 3.10.11
numpy 1.23.5
torch 2.2.1+cu121 cuda 12.1 True
onnx 1.15.0
onnxruntime 1.17.3
mlagents C:\unity-ml-agents-lab\python\.venv\lib\site-packages\mlagents\__init__.py
mlagents_envs 1.2.0.dev0
protobuf 3.20.3
tensorboard 2.20.0
```

`torch.cuda.is_available() == True` — GPU (GTX 1650) доступен для обучения.

### 6.3. `pip check` (требование 4.4)

```
> python -m pip check
No broken requirements found.
```

Конфликтов нет. Границы версий, которые этот результат обеспечивают, зафиксированы
в `python/pyproject.toml`; определяющее ограничение — `mlagents_envs`:
`numpy>=1.23.5,<1.24.0` и `protobuf>=3.6,<3.21` (источник — `ml-agents-envs/setup.py`
тега `release_23_tag`). Именно они задают выбор `onnx==1.15.0` и `onnxruntime 1.17.x`.

Точный слепок — `python/requirements.lock.txt` (55 пакетов).

### 6.4. `pytest` каркаса

```
> python -m pytest tests -q
........................................................................ [ 96%]
...                                                                      [100%]
75 passed
```

Покрыто: контракт экспорта ONNX (имена, типы, константы, opset, паритет
с PyTorch, диапазоны действий), различение `terminated` / `truncated`
в векторизованной обёртке, схема конфигов, расписания, IQM с бутстрэпом,
обязательные теги TensorBoard, структура каталога прогона.

### 6.5. Unity

| Что | Значение | Источник |
|---|---|---|
| Установленный редактор | **6000.5.4f1** (`5cb7df797b7d`) | `C:\Program Files\Unity\Hub\Editor\6000.5.4f1` |
| `ProjectVersion.txt` | приведён к 6000.5.4f1 | допущение A-20 |
| `com.unity.ml-agents` | 4.0.3 — **без изменений** | `Packages/manifest.json` |
| `com.unity.ai.inference` | 2.6.1 — **без изменений** | там же |
| URP | 17.5.0 — без изменений | там же |

Пакеты, обновлённые редактором при первом импорте, и удалённый
`com.unity.ai.assistant` — см. допущения A-21 и A-22.

### 6.6. Проверки на стороне Unity (выполнены, с выводом)

| Проверка | Команда | Результат |
|---|---|---|
| Компиляция проекта и открытие сцен | `-executeMethod ProjectBootstrap.ConfigureAndValidate` | код возврата **0**, `error CS` — 0 шт.; «сцена открыта без сбоев» для `E01_GridWorld` (4 объекта) и `E03_RollerBall` (9 объектов) |
| Валидация сцен | `-executeMethod LabRL.EditorTools.SceneValidator.ValidateAllBatch` | код возврата **1**: 2 ошибки, 2 предупреждения — см. ниже |
| Headless-сборка | `python scripts/build_env.py E01_GridWorld` | код возврата **0**, `E01_GridWorld собран за 00:06:01`, 132 685 497 байт |
| Подключение Python к билду | `labrl.envs.unity_env` + `VecUnityEnv` | подключение установлено, размерности напечатаны — см. §6.7 |

Замечания `SceneValidator` — это **реальные** несоответствия существующих сред
стандарту 7.2, а не сбой валидатора:

```
[предупреждение] E01_GridWorld: в ENV_SPEC.md нет строки-контракта <!-- validator: ... -->
[ОШИБКА]        E01_GridWorld: Agent: агент вне TrainingArea (нет TrainingAreaBase среди родителей)
[предупреждение] E03_RollerBall: в ENV_SPEC.md нет строки-контракта <!-- validator: ... -->
[ОШИБКА]        E03_RollerBall: RollerAgent: агент вне TrainingArea
```

Behavior Name, `DecisionRequester` и явный `MaxStep` проверку прошли —
переименование поведений при реструктуризации сработало. Приведение обеих сред
к стандарту 7.2 — работа Фазы 2.

### 6.7. Гейт Ф1: подключение к headless-билду

```
behavior: E01_GridWorld?team=0
наблюдения (1 сенсоров):
  obs_0: shape=(25,)  имя сенсора: 'VectorSensor_size25'
действия: непрерывных 0, дискретные ветки (4,)

арен (параллельных сред): 1
формы наблюдений батчем: [(1, 25)]
шаг   0: reward=[-0.04] terminated=[False] truncated=[False] active=[ True]
шаг  20: reward=[-1.04] terminated=[ True] truncated=[False] active=[ True]
всего завершённых эпизодов: 3
```

Что этим подтверждено: размерность наблюдения совпадает с ТЗ среды (one-hot 25),
пространство действий — одна дискретная ветка из 4, награды соответствуют
функции награды (`-0.04` за шаг, `-1.0` за ловушку), а **различение
`terminated` / `truncated` работает на живой среде**: попадание в ловушку —
истинное завершение, а не обрыв по `MaxStep`.

Отдельная находка. Unity отдаёт **полное** имя поведения `E01_GridWorld?team=0`
(`BehaviorParameters.FullyQualifiedBehaviorName`), а не `E01_GridWorld`. Обёртка
приводит идентификатор среды к полному имени (`resolve_behavior_name`), покрыто
тестами `python/tests/test_behavior_name.py`.

**Editor-режим гейта Ф1 остаётся невыполненным:** он требует нажатия Play
в открытом редакторе, то есть действия пользователя.

---

## 7. Стек перепроверен на другой машине (2026-08-20)

Работа продолжилась на машине, отличной от той, где выполнялись Фазы 1–3.
Раздел фиксирует **фактический** стек этой машины: ни одна строка не взята
из предыдущих разделов, все получены выводом команд.

### 7.1. Unity

Установленные редакторы:

```
$ ls -d "/c/Program Files/Unity/Hub/Editor/"*/
/c/Program Files/Unity/Hub/Editor/6000.5.4f1/
```

Единственный. `ProjectVersion.txt` возвращён на него (коммит `09fae88`);
до этого в файле стояла 6000.5.8f1 — версия предыдущей машины (A-20).

Проверка работоспособности — не чтением файла, а запуском:

```
$ Unity.exe -batchmode -quit -projectPath unity/MLAgentsLab \
    -executeMethod LabRL.EditorTools.SceneValidator.ValidateAllBatch

SceneValidator: ошибок 0, предупреждений 0
код возврата 0
```

Статус: `VERIFIED`.

### 7.2. Пакеты Unity

Ключевые пакеты не изменились:

| Пакет | Версия |
|---|---|
| `com.unity.ml-agents` | 4.0.3 |
| `com.unity.ai.inference` | 2.6.1 |
| `com.unity.render-pipelines.universal` | 17.5.0 |

Редактор понизил две транзитивные зависимости, поставляемые вместе с ним:
`com.unity.burst` 1.8.30 → 1.8.29, `com.unity.searcher` 4.9.5 → 4.9.4
(A-25a). Ни та, ни другая в обучении, экспорте и инференсе не участвуют.

**Версия протокола связи** сверена с обеих сторон:

```
$ grep k_ApiVersion Library/PackageCache/com.unity.ml-agents@*/Runtime/Academy.cs
const string k_ApiVersion = "1.5.0";

$ python -c "from mlagents_envs.environment import UnityEnvironment; print(UnityEnvironment.API_VERSION)"
1.5.0
```

Совпадают — Python и Unity договорятся. Статус: `VERIFIED`.

### 7.3. Python

Окружение пересоздано с нуля по `python/requirements.lock.txt`:

```
$ python/.venv/Scripts/python.exe -V
Python 3.10.12

$ python/.venv/Scripts/python.exe -m pip check
No broken requirements found.
```

Фактические версии:

| Пакет | Версия | Отличие от lock-файла |
|---|---|---|
| python | 3.10.12 | lock снят на 3.10.11; обе в диапазоне `>=3.10.1,<=3.10.12`, требуемом `mlagents_envs` |
| torch | 2.2.1+cu121 | совпадает |
| mlagents, mlagents_envs | 1.2.0.dev0 (`release_23_tag`) | совпадает |
| numpy | 1.23.5 | совпадает |
| protobuf | 3.20.3 | совпадает |
| onnx | 1.15.0 | совпадает |
| onnxruntime | 1.17.3 | совпадает |

Способ установки на этой машине отличался от A-14: git-эндпоинты GitHub
работали нормально, и `mlagents` установлен обычным клонированием тега
`release_23_tag` (около двух минут). Пофайловая загрузка не потребовалась.

### 7.4. GPU

```
$ nvidia-smi --query-gpu=name,driver_version --format=csv
NVIDIA GeForce GTX 1050 Ti, 527.37

$ python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
True NVIDIA GeForce GTX 1050 Ti
```

Другой GPU, чем на машине Фаз 1–3 (там была GTX 1650, драйвер 560.94).
Драйвер 527.37 старше, но выше минимума для CUDA 12.x (525.60), и
`torch.cuda.is_available()` это подтверждает.

Практическое замечание: на политиках учебного масштаба (MLP из двух слоёв
по 256) выигрыш GPU перед CPU невелик — узкое место не в матричных
умножениях, а в шаге среды Unity. Устройство выбирается автоматически
(`labrl.utils.seeding.resolve_device`), и принудительный CPU не ломает
ничего.

### 7.5. Тесты

```
$ python/.venv/Scripts/python.exe -m pytest python/tests
301 passed
```

Статус: `VERIFIED`.

### 7.6. Что это означает для воспроизводимости

Три вещи, важные для требования 1.4 (пользователь с чистым клоном проходит
путь без обращения к автору):

1. **Стек воспроизводится по lock-файлу на другой машине** — проверено
   фактически, а не предположено;
2. **Патч-версия Unity внутри минорной ветки не влияет** на контракт ONNX,
   компиляцию и валидацию сцен;
3. **Патч-версия Python внутри диапазона `mlagents_envs` не влияет**:
   3.10.11 и 3.10.12 дают одинаково зелёные тесты.

Чего это **не** означает: перенос на другую минорную версию Unity (6000.4,
6000.6) или Python (3.11) не проверялся и, по требованиям пакетов,
работать не обязан.
