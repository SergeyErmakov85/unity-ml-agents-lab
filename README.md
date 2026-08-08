# Unity ML-Agents Lab

Личная лаборатория проектов на [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents): от базовых обучающих примеров до кастомных сред, self-play и научных экспериментов.

Весь репозиторий — **один Unity-проект**. Каждая среда обучения с подкреплением — это папка со сценой внутри `Assets/ML-ENVIRONMENTS`: открываешь проект в Unity, открываешь нужную сцену и запускаешь свою задачу обучения.

## Структура репозитория

```
unity-ml-agents-lab/            # ← корень = Unity-проект
├── Assets/
│   ├── Editor/                      # Общие editor-утилиты (ProjectBootstrap,
│   │                                #   MLAgentsTrainingValidator)
│   ├── Settings/                    # URP-пайплайн проекта (генерируется bootstrap-ом)
│   ├── ML-Agents/                   # Общие ассеты из репозитория Unity ML-Agents
│   │   └── Examples/SharedAssets/   #   скрипты, префабы, меши, материалы примеров
│   └── ML-ENVIRONMENTS/
│       ├── 01-Basics/               # Базовые концепции ML-Agents
│       │   └── Hit_the_ball/        #   RollerAgent: докатись до цели (PPO)
│       ├── 02-Examples/             # Примеры
│       │   └── Greed_world/         #   GridWorld 5×5 для Q-learning (TS-001)
│       ├── 03-Classic-Games/        # Классические игры (Pong, Snake и т.п.)
│       ├── 04-Physics/              # Проекты с физикой (баланс, качение, метание)
│       ├── 05-MultiAgent/           # Многоагентные среды
│       ├── 06-Custom-Environments/  # Кастомные окружения
│       ├── 07-Self-Play/            # Self-play сценарии
│       ├── 08-Curriculum-Learning/  # Curriculum learning
│       ├── 09-Imitation-Learning/   # Imitation learning / behavioral cloning
│       └── 10-Research/             # Исследовательские эксперименты
├── Packages/                        # Манифест пакетов (ml-agents, URP, Input System)
├── ProjectSettings/
├── config/ml-agents-reference/      # Справочные trainer-конфиги официальных примеров
├── docs/                            # Документация (см. docs/TRAINING.md)
├── tools/env-template/              # Шаблон новой среды
├── scripts/                         # PowerShell: setup-python, train, tensorboard,
│                                    #   build-scenes, new-environment
├── requirements.txt                 # Python-зависимости (mlagents 1.1.0, torch)
└── .python-version                  # 3.10.12
```

Каждая среда содержит свои `Scenes/`, `Scripts/`, `Materials/`, editor-скрипт сборки сцены (`Editor/…Setup.cs`, меню **Tools → RL**) и `config/` с YAML для `mlagents-learn`.

## Как запустить обучение в среде

```powershell
scripts\setup-python.ps1                        # один раз: .venv c Python 3.10.12 и mlagents
scripts\train.ps1 -List                         # какие среды доступны
scripts\train.ps1 Greed_world -RunId gw-01      # запустить тренер
```

Когда в консоли появится `Listening on port 5004` — открыть сцену этой среды в Unity
и нажать **Play**. Результаты пишутся в `results/` (в git не попадает),
метрики — `scripts\tensorboard.ps1`.

Без скриптов то же самое:

```
.venv\Scripts\mlagents-learn.exe Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/config/RollerAgent.yaml --run-id=roller-01
```

Если после клонирования проект нуждается в настройке (URP, теги, список сцен) —
выполнить в редакторе **Tools → RL → Configure Project**, а готовность сред
к обучению проверить через **Tools → RL → Validate Training Setup**.

Полное руководство, ускорение обучения и разбор типовых ошибок — **[docs/TRAINING.md](docs/TRAINING.md)**.

## Новая среда

```powershell
scripts\new-environment.ps1 -Name Pendulum -Category 04-Physics
```

## Требования

- Unity 6000.5.4f1
- Пакет `com.unity.ml-agents` 4.0.3 (ставится автоматически из манифеста)
- Python 3.10.12 + `mlagents` 1.1.0 (ставит `scripts\setup-python.ps1` через [uv](https://docs.astral.sh/uv/))

## Лицензия

Проект распространяется под лицензией MIT — см. [LICENSE](LICENSE).
