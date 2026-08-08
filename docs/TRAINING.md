# Обучение агентов: полный порядок действий

Инструкция покрывает любую среду из `Assets/ML-ENVIRONMENTS`.

## Версии, на которых всё сходится

| Компонент | Версия | Где зафиксировано |
|---|---|---|
| Unity | 6000.5.4f1 | `ProjectSettings/ProjectVersion.txt` |
| `com.unity.ml-agents` | 4.0.3 | `Packages/manifest.json` |
| `com.unity.ai.inference` | 2.6.1 | зависимость ml-agents |
| Python | 3.10.12 | `.python-version` |
| `mlagents` (Python) | 1.1.0 | `requirements.txt` |
| `torch` | 2.2.x | `requirements.txt` |

Communication API у C#-пакета и Python-пакета — **1.5.0**, поэтому они
совместимы. При обновлении одной стороны проверяйте вторую: несовпадение API
даёт ошибку вида *«The communication API version is not compatible»*.

`com.unity.ml-agents` **4.0.0+** уже включает бывший пакет
`com.unity.ml-agents.extensions` — отдельно ставить его не нужно.

## 1. Один раз: Python-окружение

```powershell
scripts\setup-python.ps1
```

Скрипт через [uv](https://docs.astral.sh/uv/) поставит Python 3.10.12, создаст
`.venv` в корне репозитория и установит зависимости из `requirements.txt`.

Варианты:

```powershell
scripts\setup-python.ps1 -Cuda cu121   # torch с поддержкой GPU
scripts\setup-python.ps1 -Locked       # точные версии из requirements.lock.txt
scripts\setup-python.ps1 -Recreate     # пересоздать .venv с нуля
```

Проверка вручную:

```powershell
.venv\Scripts\mlagents-learn.exe --help
```

> `.venv/` и `results/` в git не попадают.

## 2. Один раз: настройка Unity-проекта

Открыть корень репозитория как Unity-проект и выполнить
**Tools → RL → Configure Project** (или `scripts\build-scenes.ps1 -ValidateOnly`).

Это создаёт URP-пайплайн, добавляет теги `agent/goal/trap/wall` и заполняет
список сцен сборки **всеми** сценами из `Assets/ML-ENVIRONMENTS` — новая среда
попадает туда автоматически, править список руками не нужно.

## 3. Проверить готовность среды

```
Unity → Tools → RL → Validate Training Setup
```
или в batch-режиме (**только при закрытом редакторе** — Unity не отдаёт
блокировку проекта; `build-scenes.ps1` это распознаёт и подсказывает):
```powershell
scripts\build-scenes.ps1 -ValidateOnly
```

Проверяются вещи, которые чаще всего ломают запуск:

- у каждого агента задан **Behavior Name**;
- под этот Behavior Name есть секция в `<Среда>/config/*.yaml`;
- Behavior Name уникален в пределах проекта (иначе тренер смешает агентов разных сред);
- у агента есть действия и источник наблюдений;
- присутствует **Decision Requester** (иначе решения не запрашиваются).

Список сред с их сценами, конфигами и behavior-именами:

```powershell
scripts\train.ps1 -List
```

## 4. Запуск обучения

```powershell
scripts\train.ps1 Greed_world -RunId gridworld-01
```

Скрипт сам находит `config/*.yaml` внутри папки среды. Когда в консоли появится

```
Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
```

— откройте сцену этой среды в Unity и нажмите **Play**. Обучение пойдёт.
Остановка — `Ctrl+C` в консоли: модель сохранится.

Полезные варианты:

```powershell
scripts\train.ps1 Hit_the_ball -RunId roller-01 -Resume   # продолжить прерванный запуск
scripts\train.ps1 Hit_the_ball -RunId roller-01 -Force    # перезаписать результаты
scripts\train.ps1 Hit_the_ball -RunId roller-01 -Inference # прогон без обучения
scripts\train.ps1 Hit_the_ball -- --time-scale=20 --no-graphics
```

Эквивалент без скрипта:

```powershell
.venv\Scripts\mlagents-learn.exe `
  Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/config/RollerAgent.yaml `
  --run-id=roller-01
```

## 5. Метрики

```powershell
scripts\tensorboard.ps1                 # все запуски
scripts\tensorboard.ps1 -RunId roller-01
```

Открыть <http://localhost:6006>. Главное — `Environment/Cumulative Reward`
(должна расти) и `Environment/Episode Length`.

## 6. Использовать обученную модель

1. Файл `results\<run-id>\<BehaviorName>.onnx` перетащить в проект
   (обычно в `<Среда>/Models/`).
2. В сцене на агенте: **Behavior Parameters → Model** — назначить `.onnx`.
3. **Behavior Type** переключить на **Inference Only**.
4. Нажать Play — агент работает без Python.

## Ускорение обучения

- **Несколько тренировочных зон в сцене.** Продублировать `TrainingArea`
  со смещением по X — данные собираются в несколько раз быстрее.
  У `Greed_world` префаб `TrainingArea` для этого и сделан (шаг 8 юнитов).
- **`--time-scale`** — ускоряет симуляцию (за счёт точности физики):
  `scripts\train.ps1 <Среда> -- --time-scale=20`.
- **`--no-graphics`** — отключает отрисовку (только для сред без камер-сенсоров).
- **Несколько экземпляров собранного билда**: собрать среду в `.exe`, затем
  `scripts\train.ps1 <Среда> -EnvPath build\env.exe -NumEnvs 4`.
- **`ProjectSettingsOverrides`** из `Assets/ML-Agents/Examples/SharedAssets/Scripts/`
  — компонент, через который `--time-scale` и параметры физики применяются к сцене.

## Новая среда

```powershell
scripts\new-environment.ps1 -Name Pendulum -Category 04-Physics
```

Создаётся рабочая заготовка (`Scripts/`, `Editor/`, `config/`, `README.md`).
Дальше: открыть Unity (компиляция), `scripts\build-scenes.ps1 Pendulum`,
`scripts\train.ps1 Pendulum`.

## Типовые ошибки

| Симптом | Причина и лечение |
|---|---|
| `Couldn't connect to trainer on port 5004` в Unity | `mlagents-learn` не запущен или уже завершился. Сначала консоль, потом Play. |
| Тренер ждёт вечно, Play нажат | Behavior Type = **Heuristic Only** или **Inference Only**. Нужен **Default**. |
| `The communication API version is not compatible` | Разошлись версии `com.unity.ml-agents` и Python-пакета `mlagents`. См. таблицу версий выше. |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `setuptools` 81+. В `requirements.txt` стоит пин `setuptools<81` — переустановите окружение. |
| `Behavior <name> not found in config` / агент обучается «в никуда» | Behavior Name в сцене ≠ ключ в YAML. Прогоните **Validate Training Setup**. |
| Ошибка про несовпадение размерности наблюдений | `VectorObservationSize` в Behavior Parameters ≠ числу `AddObservation` в `CollectObservations`. |
| Агент не двигается, эпизоды не идут | Нет **Decision Requester** и нет ручного `RequestDecision()`. |
| Материалы примеров пурпурные | Ассеты ML-Agents под Built-in RP. **Window → Rendering → Render Pipeline Converter**. |
| Управление с клавиатуры не работает в Heuristic | `Active Input Handling` должен остаться **Both** (`activeInputHandler: 2`). |
| `Unity.exe -batchmode` мгновенно выходит с кодом 1 и пустым логом | Проект открыт в редакторе (`Temp/UnityLockfile`). Закройте Unity или используйте меню `Tools → RL → …`. |

## Ссылки

- Документация пакета: <https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/index.html>
- Параметры trainer-конфига: <https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Training-Configuration-File.html>
- Справочные конфиги примеров: `config/ml-agents-reference/`
- Общие ассеты примеров: `Assets/ML-Agents/README.md`
