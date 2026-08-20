# 00_AUDIT — Аудит Фазы 0

**Дата аудита:** 2026-08-15
**Ветка:** `feature/lab-bootstrap`
**Исполнитель:** Claude Code
**Основание:** `CLAUDE_Unity-ml-agents-lab.md`, раздел 6 «Фаза 0», раздел 17.

Все утверждения ниже подкреплены выводом команд или чтением файлов. Утверждения,
которые проверить не удалось, вынесены в раздел 6 «Блокеры» со статусом `NOT VERIFIED`.

---

## 1. Обход репозиториев

### 1.1. `C:\cyber-unity-learn` (READ-ONLY)

Тип: веб-приложение (Vite + React + TypeScript + Tailwind + Supabase), сайт
`https://rl-cuber-unity-code.com/`. Уроки — это **страницы React**, а не markdown-файлы.

Дерево (глубина 3, без `node_modules/`, `dist/`, `.git/`, `.lovable/`):

```
cyber-unity-learn/
├── .claude/skills/add-math-lecture/
├── .cursor/rules/
├── .github/workflows/
├── @docs/audit/                  # LOVABLE_STYLE_GUIDE.md, racing-guide.md.md (131 КБ), lesson-1-5-*
├── public/{images,og}/
├── scripts/
├── src/
│   ├── assets/videos/
│   ├── components/               # auth, knowledge-map, landing, lesson-1-2, lesson-2-6,
│   │                             # lesson-3-1…3-8, lessons, math-rl, project-2,
│   │                             # racing-agent, textbook, ui
│   ├── config/                   # openAccess.ts, crosslinks.ts (77 КБ)
│   ├── content/                  # hubs.ts, learningMap.ts, knowledgeMap.ts,
│   │                             # lessonContextLinks.ts, mathMindMap.ts,
│   │                             # math-textbook/ (7 частей), textbook/
│   ├── data/lessons.ts           # реестр LessonMeta
│   ├── hooks/, integrations/supabase/, lib/mcp/, pages/, styles/
└── supabase/{functions,migrations}/
```

Ключевые источники истины по составу курса:

| Файл | Что содержит |
|---|---|
| `src/content/learningMap.ts` | 3 ступени, 21 урок + 3 проекта, слаги и маршруты |
| `src/data/lessons.ts` | метаданные части уроков (уровень, PRO/FREE, теги) |
| `src/App.tsx` | фактические маршруты (95 путей) — доказательство, что страница существует |
| `src/pages/UnityProjectsHub.tsx` | 6 Unity-проектов курса с указанием алгоритма |
| `src/content/math-textbook/` | математический учебник, 7 частей (не курс RL, но источник терминологии) |

### 1.2. `C:\unity-ml-agents-lab` (READ / WRITE)

Тип: **единый Unity-проект в корне репозитория** (Unity 6000.5.4f1, URP,
`com.unity.ml-agents` 4.0.3). Git: ветка `main`, remote
`https://github.com/SergeyErmakov85/unity-ml-agents-lab.git`, 8 коммитов,
последний — `c35e558 Единый Unity-проект в корне`.

Полный перечень значимых файлов (без `.meta`, `Library/`, `Temp/`, `.git/`):

```
Assets/
  Editor/ProjectBootstrap.cs                                   3 668 Б
  Settings/{URP_Asset.asset, URP_Renderer.asset}
  DefaultVolumeProfile.asset, UniversalRenderPipelineGlobalSettings.asset
  ML-ENVIRONMENTS/
    01-Basics/Hit_the_ball/
      Editor/RLEnvironmentSetup.cs                             5 344 Б
      Materials/{Agent,Target,Floor,Wall}Mat.mat
      Scenes/RLTrainingScene.unity                            33 414 Б
      Scripts/RollerAgent.cs                                   4 406 Б
      config/RollerAgent.yaml                                    511 Б
    02-Examples/Greed_world/
      Editor/GridWorldSetup.cs                                16 484 Б
      INSTRUCTIONS.md  (ТЗ TS-001)                            28 404 Б
      Materials/Mat_{Agent,Goal,Ground,GridLines,Start,Trap,Wall}.mat
      Prefabs/{TrainingArea.prefab (34 035 Б), Wall.prefab, TrapMarker.prefab}
      Scenes/GridWorld.unity                                  39 867 Б
      Scripts/{GridWorldAgent.cs, GridWorldEnvironment.cs, GridWorldUI.cs}
      Textures/Tex_GridLines.png
    03-Classic-Games/ … 10-Research/                          пусто (.gitkeep)
Packages/{manifest.json, packages-lock.json}
ProjectSettings/*                                             27 файлов
docs/, scripts/, tools/                                       пусто (.gitkeep)
.gitignore (2 404 Б), .gitattributes (5 083 Б), README.md, CLAUDE.md, LICENSE
CLAUDE_Unity-ml-agents-lab.md                                50 844 Б (untracked)
```

Даты изменения всех файлов: 2026-08-14 (кроме `.claude/settings.local.json` — 2026-08-15).
Каталог `Library/` **отсутствует** — проект ни разу не открывался на этой машине
(либо `Library/` был удалён).

---

## 2. Состав уроков `cyber-unity-learn`

Извлечено из `src/content/learningMap.ts`, `src/App.tsx` (маршруты), заголовков секций
в `src/pages/CourseLesson*.tsx` и `src/components/lesson-3-*/`, а также поиском по
ключевым словам методов RL во всех `src/pages/*.tsx`.

### 2.1. Ступень 1 «Новичок» (FREE, 6 недель)

| № | Маршрут | Название | Методы RL, фактически разобранные в уроке |
|---|---|---|---|
| 1.1 | `/courses/1-1` | Что такое RL? | Цикл RL, exploration/exploitation, формализм MDP, мини-GridWorld, CartPole |
| 1.2 | `/courses/1-2` | Установка окружения: PyTorch + Unity ML-Agents | Anaconda, conda-среда Python 3.10, PyTorch+CUDA, Unity 6, **ML-Agents Release 22 через git**, VS Code + Jupyter |
| 1.3 | `/courses/1-3` | Марковские процессы принятия решений | MDP `(S, A, T, R, γ)`, марковское свойство, π, `G_t`, `V^π`, `Q^π`, **Value Iteration**, вывод уравнения Беллмана |
| 1.4 | `/courses/1-4` | Q-Learning: табличный метод | **Табличный Q-learning**, TD-обновление, ε-greedy, среда FrozenLake |
| 1.5 | `/courses/1-5` | CartPole — первый RL-агент | Gymnasium, случайный baseline, Q-learning с дискретизацией, ε-greedy, агент на PyTorch |
| 1.6 | `/courses/1-6` | DQN с нуля на PyTorch | **DQN**, Replay Buffer, Target Network, Huber Loss, gradient clipping, сохранение модели |
| 1.7 | `/courses/1-7` | Exploration vs Exploitation | **Многорукие бандиты**: ε-greedy, **UCB**, **Thompson Sampling**, сравнение стратегий |
| П-1 | `/courses/project-1` | Проект-1: «Баланс в 3D» | Unity-среда + «мозг» на PyTorch + цикл RL + метрики |

### 2.2. Ступень 2 «Средний» (PRO, 6 недель)

| № | Маршрут | Название | Методы RL |
|---|---|---|---|
| 2.1 | `/courses/2-1` | Policy Gradient и теорема градиента | **REINFORCE**, вывод `∇J(θ)`, проблема дисперсии, **baseline** |
| 2.2 | `/courses/2-2` | PPO — реализация с нуля | **PPO** (clipped objective), критик и **GAE**, сравнение PPO vs DQN, упоминание TRPO |
| 2.3 | `/courses/2-3` | Непрерывные действия и Actor-Critic | **Actor-Critic** для непрерывных действий, гауссова политика, PPO vs SAC |
| 2.4 | `/courses/2-4` | Reward Shaping | Sparse vs dense, credit assignment, **potential-based shaping (PBRS)**, ловушки, **ICM/RND** (упоминание), реализация в Unity C# |
| 2.5 | `/courses/2-5` | Параллельные среды | Векторизация сред, sample efficiency, несколько `TrainingArea` |
| 2.6 | `/courses/2-6` | TensorBoard и W&B | Метрики обучения, диагностика по графикам, альтернативы |
| П-2 | `/courses/project-2` | Проект: 3D-охотник | PPO, RayPerceptionSensor, GAE, PBRS, reward hacking |
| П-3 | `/courses/project-3` | Проект: Гоночный агент | Сенсоры, логика агента C#, награды, алгоритмы, TensorBoard |

### 2.3. Ступень 3 «Продвинутый» (PRO, 8 недель)

| № | Маршрут | Название | Методы RL |
|---|---|---|---|
| 3.1 | `/courses/3-1` | SAC — Soft Actor-Critic | **SAC**: max-entropy цель, soft Bellman, soft policy iteration, два Q-критика, reparam-trick, автоподстройка α, replay buffer, SAC vs PPO |
| 3.2 | `/courses/3-2` | MA-POCA и Self-Play | **MA-POCA**, Dec-POMDP, CTDE, counterfactual baseline, posthumous credit assignment, **Self-Play**, ELO |
| 3.3 | `/courses/3-3` | Учебный план и рандомизация среды | **Curriculum Learning**, **Domain Randomization**, **ADR**, **PLR**, **UED** |
| 3.4 | `/courses/3-4` | Imitation Learning: BC и GAIL | **Behavioral Cloning**, **IRL**, **GAIL**, **AIRL**, пайплайн демонстраций |
| 3.5 | `/courses/3-5` | Деплой: ONNX-экспорт и интеграция | **ONNX**, Unity Inference Engine, чекпойнты, режимы Behavior Type, сборка билда, диагностика рассинхрона |
| 3.6 | `/courses/3-6` | Оптимизация гиперпараметров | HPO, grid/random search, байесовская оптимизация, **TPE**, прунинг (SHA/ASHA/Hyperband), Optuna, W&B Sweeps |
| 3.7 | `/courses/3-7` | Архитектуры нейросетей для RL | MLP, CNN, **LSTM/POMDP**, внимание, общий/раздельный backbone, `network_settings`, Decision Transformer |
| 3.8 | `/courses/3-8` | Финальный проект | Сквозной пайплайн: среда → награда → обучение → оптимизация → деплой → геймплей |

### 2.4. Сопутствующие разделы сайта (не уроки курса, но источник сред и терминологии)

- **Хабы** (`src/content/hubs.ts`): `pytorch`, `unity-ml-agents`, `deep-rl`, `project`, `math-rl`, `fca-rl`.
- **Алгоритмические модули**: `/algorithms/dqn`, `/algorithms/ppo`, `/algorithms/sac`, `/algorithms/a3c`
  (`DQNModule.tsx` дополнительно содержит **Double DQN**, **Dueling DQN**).
- **Unity-проекты** (`/unity-projects`, `UnityProjectsHub.tsx`):

  | id | Название | Действия | Алгоритм | Статус на сайте |
  |---|---|---|---|---|
  | `taxi-v3` | Taxi-v3: Q-Learning | Дискретные | Q-Learning / REINFORCE | ready |
  | `food-collector` | FoodCollector: REINFORCE | Гибридные | REINFORCE | flagship |
  | `ball-balance` | 3D Ball Balance | Непрерывные | PPO | ready |
  | `gridworld` | GridWorld | Дискретные | PPO | ready |
  | `soccer` | Soccer (Multi-Agent) | Смешанные | MAPOCA | ready |
  | `racing` | Racing Car | Непрерывные | SAC | ready |

- **Прочее**: `/projects/frozen-lake`, `/labs` (лаборатории бандитов: ε-greedy, UCB, Boltzmann),
  `/fca-rl` (исследовательское направление FCA), `/math-rl/textbook` (7 частей математики),
  8 статей блога (GridSensor, ONNX+Sentis, parallel envs, MA-POCA, PPO vs SAC, REINFORCE vs PPO, top-5 ошибок, Jupyter→Unity).

### 2.5. Методы, которых в уроках **нет** (проверено поиском по всем `src/pages/*.tsx`)

`SARSA` (упомянут только в `MathRL.tsx`), `Expected SARSA`, `Dyna-Q`, `Policy Iteration`
(упомянут вскользь в 3.2), `n-step`, `DDPG`, `Rainbow`, `C51`, `QR-DQN`, `Prioritized
Experience Replay` (как алгоритм; слово `PER` в выдаче — ложные срабатывания на
русских словах), `A2C`/`A3C` (только в отдельном модуле `/algorithms/a3c`, не в курсе),
`TD3` (только упоминание в 3.1).

Вывод: **референсная таксономия раздела 13.3 инструкции шире фактического курса.**
Это расхождение — предмет вопроса Q2 (см. `PLAN.md`).

---

## 3. Инвентаризация `unity-ml-agents-lab`

### 3.1. Среда `01-Basics/Hit_the_ball` (RollerAgent)

| Параметр | Значение | Источник |
|---|---|---|
| Behavior Name | `RollerAgent` | `Scenes/RLTrainingScene.unity:759` |
| Наблюдения | 8 (позиция цели 3 + позиция агента 3 + скорость x,z 2) | `Scripts/RollerAgent.cs:54-61` |
| Действия | Continuous 2 (сила по X, Z) | `Scripts/RollerAgent.cs:63-69` |
| Награды | `+1.0` при `distance < 1.42`; эпизод обрывается при `y < 0` без награды | `Scripts/RollerAgent.cs:71-82` |
| Heuristic | Есть, legacy `Input.GetAxis` | `Scripts/RollerAgent.cs:86-91` |
| Конфиг тренера | `config/RollerAgent.yaml`, PPO, `max_steps: 500000` | файл |
| Сборка сцены | `RLEnvironmentSetup.BuildTrainingScene` | `Editor/RLEnvironmentSetup.cs` |
| ТЗ (`ENV_SPEC.md`) | **отсутствует** | — |
| Обученная модель | **отсутствует** | — |

Замечания: `MaxStep` в коде не задан; нет `EnvironmentParametersChannel`;
нет `StatsRecorder`; одна арена (не параметризовано); сборка headless невозможна
(нет `BuildScript.cs`).

### 3.2. Среда `02-Examples/Greed_world` (GridWorld 5×5)

| Параметр | Значение | Источник |
|---|---|---|
| Behavior Name | `GridWorldQLearning` | `Prefabs/TrainingArea.prefab:379` |
| Наблюдения | one-hot 25 (`cols*rows`) | `Scripts/GridWorldAgent.cs:37-43` |
| Действия | Discrete, 1 ветвь × 4 (N/S/E/W) | `Scripts/GridWorldAgent.cs:45-56` |
| Награды | `stepReward` + `goalReward` / `trapReward` (значения в `GridWorldEnvironment`) | `Scripts/GridWorldAgent.cs:59-72` |
| Heuristic | Есть, поддерживает и Input System, и legacy Input | `Scripts/GridWorldAgent.cs:79-109` |
| Внешний Q-learning | `CurrentStateIndex` 0–24 открыт наружу | `Scripts/GridWorldAgent.cs:25` |
| ТЗ | `INSTRUCTIONS.md` — **TS-001, формат Playbook v3.0** (разделы 0–14) | файл |
| Конфиг тренера | **отсутствует** | — |
| Обученная модель | **отсутствует** | — |

Замечания: `TrainingArea` — Prefab (соответствует 7.2); нет
`EnvironmentParametersChannel`, нет `StatsRecorder`, нет `SceneValidator`.

### 3.3. Общая инфраструктура

| Компонент | Состояние |
|---|---|
| `Assets/Editor/ProjectBootstrap.cs` | Есть: URP-ассет, теги `agent/goal/trap/wall`, build-scene list, smoke-открытие сцен. Метод `ProjectBootstrap.ConfigureAndValidate` |
| `Assets/Shared/` (Core, Debug, Editor, Materials, Prefabs) | **отсутствует** |
| `BuildScript.cs` (headless-сборка) | **отсутствует** |
| `SceneValidator.cs` | **отсутствует** |
| Python-пакет `labrl` | **отсутствует** |
| Ноутбуки | **отсутствуют** |
| `configs/`, `builds/`, `results/`, `_archive/` | **отсутствуют** |
| `docs/`, `scripts/`, `tools/` | существуют, но **пустые** (только `.gitkeep`) |
| `.gitignore` | Покрывает `Library/`, `Temp/`, `Builds/`, `results/`, `.venv/`, `*.onnx.meta`. **Не покрывает** `builds/` в нижнем регистре как отдельную запись — фактически покрывает через `/[Bb]uilds/` |
| `.gitattributes` | Git LFS для бинарных ассетов; `git-lfs/3.7.0` установлен |

### 3.4. Что дублируется / что сломано

- **Дублирования не обнаружено.** Каждая среда живёт в своей папке, общих файлов нет.
- **Сломанного кода не обнаружено**, но и **не проверено компиляцией**: Unity версии
  `6000.5.4f1` на машине нет (см. блокер B-1), проект не открывался.
- Расхождение с целевой структурой раздела 5 инструкции: текущая раскладка —
  `Assets/ML-ENVIRONMENTS/<NN-Category>/<Name>/`, целевая —
  `unity/MLAgentsLab/Assets/Envs/E##_<Name>/`. Требуется полная реструктуризация
  (Фаза 1, через `git mv`).
- Behavior Names (`RollerAgent`, `GridWorldQLearning`) **не соответствуют** правилу 5.3
  (`E##_<Name>`). Переименование обязательно.

---

## 4. Верификация стека

Полностью изложена в `docs/01_STACK.md`. Краткое резюме:

| Компонент | Заявлено | Фактически | Статус |
|---|---|---|---|
| Unity | 6000.5.4f1 (`ProjectVersion.txt`) | не установлен; есть 2021.3.45f1, 2023.2.13f1, 2023.2.20f1, 6000.0.35f1, 6000.3.10f1, 6000.5.0a8 | **BLOCKED (B-1)** |
| `com.unity.ml-agents` | 4.0.3 (`manifest.json:9`) | манифест прочитан; пакет не разрешён (нет `Library/PackageCache`) | NOT VERIFIED |
| `com.unity.ai.inference` | 2.6.1 (`manifest.json:4`) | то же | NOT VERIFIED |
| URP | 17.5.0 (`manifest.json:12`) | то же | NOT VERIFIED |
| Python | 3.10.x (требование `mlagents`) | 3.11.4 (`Python311`), 3.14 зарегистрирован, но `C:\Python314\python.exe` не запускается | **BLOCKED (B-2)** |
| `mlagents`, `mlagents-envs` | Release 22+ | **не установлены** | **BLOCKED (B-2)** |
| `torch`, `onnx`, `onnxruntime`, `tensorboard` | требуются | **не установлены** | **BLOCKED (B-2)** |
| `numpy` | совместимая | 1.25.1 (несовместима: `mlagents` требует `<1.24.0`) | конфликт |
| conda | ожидается по уроку 1.2 | **не установлена** | расхождение с уроком |

---

## 5. Контракт ONNX

Извлечён и зафиксирован в `docs/04_ONNX_CONTRACT.md`.

**Важно:** требование инструкции 10.2 — извлечь контракт из **установленного** пакета
`mlagents`. Пакет не установлен (блокер B-2), поэтому контракт извлечён из исходников
официального репозитория по тегу `release_23_tag` (тот же тег, что соответствует
C#-пакету `com.unity.ml-agents` 4.0.0; в проекте стоит 4.0.3). Контракт помечен
статусом **DRAFT** и подлежит обязательной повторной сверке с установленным пакетом
в Фазе 1.

---

## 6. Блокеры

| # | Блокер | Последствие | Что требуется |
|---|---|---|---|
| **B-1** | Unity 6000.5.4f1 не установлен | Невозможны: открытие проекта, компиляция, `SceneValidator`, headless-сборка, проверка инференса ONNX в Unity (DoD 14) | Решение пользователя: установить 6000.5.4f1 либо перевести проект на установленную версию |
| **B-2** | Python-стек RL отсутствует целиком; доступен только Python 3.11.4, тогда как `mlagents`/`mlagents_envs` требуют `>=3.10.1,<=3.10.12` | Невозможны: обучение, экспорт ONNX, верификация, `pip check`, извлечение контракта из установленного пакета | Установить Python 3.10.x и создать `python/.venv` (Фаза 1) |
| **B-3** | Контракт ONNX не подтверждён по установленному пакету (следствие B-2) | `docs/04_ONNX_CONTRACT.md` имеет статус DRAFT | Снять после B-2 |
| **B-4** | Расхождение таксономии инструкции (13.3, 14 сред) и фактического курса (21 урок + 3 проекта, 6 Unity-проектов) | Неопределён итоговый перечень примеров | Решение пользователя (вопрос Q2 в `PLAN.md`) |

Запрет 0.3 инструкции («до завершения Фазы 0 запрещено устанавливать пакеты»)
соблюдён: ничего не устанавливалось, ничего не переносилось, ничего не удалялось.
В `C:\cyber-unity-learn` не выполнено ни одной операции записи.

---

## 7. Изменения, внесённые в ходе Фазы 0

| Действие | Обоснование |
|---|---|
| Создана ветка `feature/lab-bootstrap` от `main` | Требование 3.3 |
| Созданы `docs/00_AUDIT.md`, `docs/01_STACK.md`, `docs/02_LESSON_MAP.md`, `docs/04_ONNX_CONTRACT.md`, `docs/ASSUMPTIONS.md`, `PLAN.md` | Требование 6 / Фаза 0, п. 6 |

Структура каталогов, сцены, скрипты и настройки проекта **не изменялись**.
