# Unity ML-Agents Lab

Учебно-исследовательская лаборатория обучения с подкреплением: среды на
[Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) и **собственное
ядро обучения на PyTorch**.

Ключевое отличие от типового проекта на ML-Agents: агенты обучаются **своим кодом** —
авторскими сетями и самостоятельно реализованными алгоритмами (`python/labrl`),
а не командой `mlagents-learn`. Штатный тренер сохранён отдельно, как эталон
для сравнения «моя реализация ↔ эталон».

Каждый пример доводится до конца: сцена → headless-сборка → ноутбук обучения →
метрики в TensorBoard → веса в ONNX → **инференс обратно в Unity**.

> **Статус:** двенадцать сред `E00`–`E11` собраны, валидированы и готовы
> к обучению. Восемь из них (`E00`–`E07`) обучены и доведены до инференса
> ONNX в Unity; четыре (`E08`–`E11`) готовы к запуску — обучение на них
> запускает пользователь. Подробности — в [`PLAN.md`](PLAN.md).

## Среды

Статусы: **DONE** — обучено и проверено инференсом в Unity;
**READY** — сцена, билд, конфиг и ноутбук готовы, обучение за пользователем.

| Пример | Задача | Действия | Алгоритмы | Урок | Статус |
|---|---|---|---|---|---|
| [`E00_Bandit`](docs/envs/E00_Bandit.md) | многорукий бандит | Discrete 1×5 | ε-greedy, UCB1, Thompson | 1.7 | **DONE** |
| [`E01_GridWorld`](docs/envs/E01_GridWorld.md) | дойти до цели в сетке 5×5 | Discrete 1×4 | Value Iteration, табличный Q-learning | 1.3, 1.4 | **DONE** |
| [`E02_CartPoleUnity`](docs/envs/E02_CartPoleUnity.md) | удержать шест на тележке | Discrete 1×2 | Q-learning с дискретизацией | 1.5 | **DONE** |
| [`E03_RollerBall`](docs/envs/E03_RollerBall.md) | докатиться шаром до цели | Discrete 1×4 | DQN (Double) | 1.6 | **DONE** |
| [`E04_BallBalance`](docs/envs/E04_BallBalance.md) | удержать шар на платформе | Continuous 2 | A2C, PPO | 2.3, проект-1 | **DONE** |
| [`E05_FoodCollector`](docs/envs/E05_FoodCollector.md) | собрать еду, обходя вредную | гибридные | REINFORCE + baseline | 2.1, проект-2 | **DONE** |
| [`E06_Hunter3D`](docs/envs/E06_Hunter3D.md) | догнать цель, обходя препятствия | Continuous 2 | PPO + GAE + PBRS | 2.2, 2.4, проект-2 | **DONE** |
| [`E07_RacingCar`](docs/envs/E07_RacingCar.md) | проехать круг по трассе | Continuous 2 | SAC | 3.1, проект-3 | **DONE** |
| [`E08_SoccerArena`](docs/envs/E08_SoccerArena.md) | 2 × 2 футбол | Discrete 3×3×3 | **MA-POCA + Self-Play** | 3.2 | READY |
| [`E09_CurriculumMaze`](docs/envs/E09_CurriculumMaze.md) | лабиринт растущей сложности | Discrete 1×4 | **PPO + Curriculum + DR** | 3.3 | READY |
| [`E10_Imitation`](docs/envs/E10_Imitation.md) | коридор, где RL буксует | Discrete 1×4 | **BC, GAIL** | 3.4 | READY |
| [`E11_Research`](docs/envs/E11_Research.md) | ключ и дверь | Discrete 1×4 | **PPO + слой понятий FCA** | `/fca-rl` | READY |

Сводка результатов с IQM и доверительными интервалами — [`docs/RESULTS.md`](docs/RESULTS.md).

## Как запустить обучение

Каждая среда запускается одной командой. Сначала — смоук-тест конвейера
(сокращённый бюджет, минуты), потом полный прогон на трёх сидах:

```powershell
$py = ".\python\.venv\Scripts\python.exe"

# 1. Собрать среду (один раз на среду)
& $py scripts\build_env.py E08_SoccerArena

# 2. Смоук-тест: проверяет весь конвейер, критерий приёмки не проверяет
& $py scripts\train.py --config configs\E08_SoccerArena__mapoca.yaml --seed 0 --quick

# 3. Полное обучение на трёх сидах
& $py scripts\train.py --config configs\E08_SoccerArena__mapoca.yaml --all-seeds

# 4. Метрики
.\scripts\tb.ps1

# 5. Проверка инференса в Unity на 20 эпизодах (требование 10.6)
& $py scripts\check_inference.py --config configs\E08_SoccerArena__mapoca.yaml --python-reward <оценка>
```

Проверить **все** конвейеры разом, не обучая ничего по-настоящему:

```powershell
& $py scripts\smoke_all.py --list     # что будет запущено и где нет билда
& $py scripts\smoke_all.py            # все 18 конфигов, сокращённый бюджет
```

Конфиги всех сред — в [`configs/`](configs/); у среды с несколькими методами
их несколько (например `E10_Imitation__bc.yaml`, `__gail.yaml`,
`__ppo_discrete.yaml` — для сравнения методов на одной задаче).

То же самое, но с пояснениями и графиками, — в ноутбуках
[`notebooks/`](notebooks/), по одному на пару «среда + алгоритм».

---

## Структура репозитория

```
unity-ml-agents-lab/
├── CLAUDE_Unity-ml-agents-lab.md   # исполняемая инструкция верхнего уровня
├── PLAN.md                         # план по фазам и текущее состояние
├── docs/                           # аудит, стек, соглашения, контракт ONNX, ADR, карточки
├── unity/MLAgentsLab/              # ЕДИНСТВЕННЫЙ Unity-проект на все среды
│   └── Assets/
│       ├── Shared/                 # общий код: AgentBase, TrainingAreaBase, BuildScript…
│       └── Envs/E##_<Name>/        # среда: Scenes, Scripts, Editor, Prefabs, Models, ENV_SPEC.md
├── python/
│   ├── labrl/                      # ядро: envs, nets, algos, buffers, export, logging, eval
│   ├── tests/                      # pytest: контракт ONNX, формы, terminated vs truncated
│   └── .venv/                      # изолированное окружение (Python 3.10.x)
├── notebooks/                      # по одному ноутбуку на пример
├── configs/                        # наши конфиги + configs/mlagents/ (эталон штатного тренера)
├── scripts/                        # build_env.py, verify_onnx.py, tb.ps1
├── builds/                         # headless-сборки (в .gitignore)
├── results/                        # прогоны обучения (в .gitignore)
└── _archive/                       # вытесненные материалы; безвозвратно ничего не удаляется
```

## Требования

| Что | Версия | Проверка |
|---|---|---|
| Unity | **6000.5.4f1** | `unity/MLAgentsLab/ProjectSettings/ProjectVersion.txt` |
| `com.unity.ml-agents` | 4.0.3 | `unity/MLAgentsLab/Packages/manifest.json` |
| Inference Engine (`com.unity.ai.inference`) | 2.6.1 | там же |
| Python | **3.10.x** (не 3.11+: `mlagents-envs` требует `>=3.10.1,<=3.10.12`) | `py -0p` |
| PyTorch | 2.2.1 + CUDA 12.1 | `python/requirements.lock.txt` |
| `mlagents`, `mlagents-envs` | из git, тег `release_23_tag` | `docs/01_STACK.md` |

Полный список с выводами команд верификации — [`docs/01_STACK.md`](docs/01_STACK.md).

## Быстрый старт

### 1. Unity

Установить через Unity Hub редактор **6000.5.4f1** и открыть проект
`unity/MLAgentsLab`. Если проект открыт впервые, выполнить в редакторе
**Tools → RL → Configure Project** — создаст URP-пайплайн, теги и список сцен.

### 2. Python

```powershell
py -3.10 -m venv python\.venv
$py = ".\python\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

# torch с CUDA доступен только в индексе PyTorch, не на PyPI
& $py -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# mlagents нужной версии на PyPI нет — ставится из git по тегу release_23_tag
& $py -m pip install "git+https://github.com/Unity-Technologies/ml-agents.git@release_23_tag#subdirectory=ml-agents-envs" --no-deps
& $py -m pip install "git+https://github.com/Unity-Technologies/ml-agents.git@release_23_tag#subdirectory=ml-agents" --no-deps

& $py -m pip install -r python\requirements.lock.txt
& $py -m pip install -e python --no-deps
& $py -m pip check
& $py -m pytest python\tests -q
```

Зелёный `pytest` означает, что ядро и контракт экспорта ONNX работают.
Точный слепок окружения — [`python/requirements.lock.txt`](python/requirements.lock.txt);
там же процедура воспроизведения и причины ручных правок двух строк.

### 3. Проверка связи с Unity

Открыть `notebooks/00_setup_check.ipynb` и выполнить сверху вниз: ноутбук
подключается к редактору и к headless-сборке и печатает размерности пространств.

## Сквозной цикл одного примера

```powershell
# 1. Проверить сцены
python scripts\build_env.py --validate-only

# 2. Собрать среду без открытия редактора
python scripts\build_env.py E03_RollerBall

# 3. Обучить — ноутбук notebooks/E03_RollerBall__dqn.ipynb (QUICK_RUN=True для проверки)

# 4. Посмотреть метрики
.\scripts\tb.ps1

# 5. Проверить экспортированную модель
python scripts\verify_onnx.py results\...\onnx\policy.onnx --discrete-branches 4
```

Затем модель кладётся в `Assets/Envs/E##_<Name>/Models/`, назначается в
`BehaviorParameters.Model`, тип поведения — `Inference Only`. Критерий приёмки:
средняя награда в Unity ≥ 0.8 от Python-оценки на 20 эпизодах.

Подробно — [`docs/06_WORKFLOW.md`](docs/06_WORKFLOW.md).

## Документация

| Документ | О чём |
|---|---|
| [`docs/00_AUDIT.md`](docs/00_AUDIT.md) | аудит репозиториев и уроков |
| [`docs/01_STACK.md`](docs/01_STACK.md) | версии стека с выводами команд |
| [`docs/02_LESSON_MAP.md`](docs/02_LESSON_MAP.md) | карта «урок → пример → алгоритм → статус» |
| [`docs/03_CONVENTIONS.md`](docs/03_CONVENTIONS.md) | имена, каталоги, коды примеров |
| [`docs/04_ONNX_CONTRACT.md`](docs/04_ONNX_CONTRACT.md) | контракт экспорта ONNX для Unity |
| [`docs/05_TENSORBOARD.md`](docs/05_TENSORBOARD.md) | обязательная схема метрик |
| [`docs/06_WORKFLOW.md`](docs/06_WORKFLOW.md) | сквозной цикл разработки примера |
| [`docs/07_TROUBLESHOOTING.md`](docs/07_TROUBLESHOOTING.md) | измеренные грабли и их разбор |
| [`docs/RESULTS.md`](docs/RESULTS.md) | сводка результатов всех примеров |
| [`docs/envs/`](docs/envs/) | карточка на каждую среду |
| [`docs/algos/`](docs/algos/) | карточка на каждый алгоритм |
| [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) | реестр допущений |
| [`docs/adr/`](docs/adr/) | архитектурные решения |
| [`docs/feedback_to_lessons.md`](docs/feedback_to_lessons.md) | предложения по урокам курса (правки в `cyber-unity-learn` не вносятся — правило 3.1) |

## Лицензия

MIT — см. [LICENSE](LICENSE).
