# 01_STACK — Верификация технологического стека

**Дата:** 2026-08-15
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
