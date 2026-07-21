# Unity ML-Agents Lab

Личная лаборатория проектов на [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents): от базовых обучающих примеров до кастомных сред, self-play и научных экспериментов.

Весь репозиторий — **один Unity-проект**. Каждая среда обучения с подкреплением — это папка со сценой внутри `Assets/ML-ENVIRONMENTS`: открываешь проект в Unity, открываешь нужную сцену и запускаешь свою задачу обучения.

## Структура репозитория

```
unity-ml-agents-lab/            # ← корень = Unity-проект
├── Assets/
│   ├── Editor/                      # Общие editor-утилиты (ProjectBootstrap)
│   ├── Settings/                    # URP-пайплайн проекта (генерируется bootstrap-ом)
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
├── docs/                            # Документация и заметки
├── tools/                           # Вспомогательные утилиты
└── scripts/                         # Скрипты обучения/автоматизации
```

Каждая среда содержит свои `Scenes/`, `Scripts/`, `Materials/`, editor-скрипт сборки сцены (`Editor/…Setup.cs`, меню **Tools → RL**) и при необходимости `config/` с YAML для `mlagents-learn`.

## Как запустить обучение в среде

1. Открыть репозиторий как Unity-проект (корневую папку).
2. Открыть сцену нужной среды, например
   `Assets/ML-ENVIRONMENTS/02-Examples/Greed_world/Scenes/GridWorld.unity`.
3. В терминале запустить тренер, указав конфиг среды:

   ```
   mlagents-learn Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/config/RollerAgent.yaml --run-id=roller-01
   ```

4. Нажать Play в редакторе — начнётся обучение. Результаты пишутся в `results/` (в git не попадает).

Если после клонирования проект нуждается в настройке (URP, теги, список сцен) — выполнить в редакторе **Tools → RL → Configure Project**.

## Требования

- Unity 6000.5 (LTS)
- Пакет `com.unity.ml-agents` 4.0.3 (ставится автоматически из манифеста)
- Python 3.10+ с пакетом `mlagents`

## Лицензия

Проект распространяется под лицензией MIT — см. [LICENSE](LICENSE).
