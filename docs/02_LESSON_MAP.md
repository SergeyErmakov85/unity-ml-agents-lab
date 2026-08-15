# 02_LESSON_MAP — Карта «урок → пример → алгоритм»

**Статус документа:** `ЧЕРНОВИК` — требует подтверждения пользователя (гейт Ф0).
**Дата:** 2026-08-15
**Источник истины:** фактическое содержимое `C:\cyber-unity-learn`
(`src/content/learningMap.ts`, `src/App.tsx`, страницы `src/pages/*.tsx`) — см. `docs/00_AUDIT.md`, раздел 2.
**Базовый URL уроков:** `https://rl-cuber-unity-code.com`

Статусы: `TODO` — пример запланирован, не начат; `WIP` — в работе;
`DONE` — все пункты DoD (раздел 14 инструкции) выполнены и подтверждены;
`MISSING_EXAMPLE` — урок есть, примера нет и не запланирован;
`NO_LESSON` — пример есть, урока нет; `CROSS` — урок сквозной, реализуется как
свойство всех примеров, а не как отдельный пример.

---

## 1. Основная карта

| Урок (маршрут) | Название | Метод | Пример `E##` | Среда | Ноутбук | Статус |
|---|---|---|---|---|---|---|
| `/courses/1-1` | Что такое RL? | Понятия MDP, exploration/exploitation | — | — | — | `CROSS` (теория) |
| `/courses/1-2` | Установка окружения | Стек PyTorch + ML-Agents | — | — | `00_setup_check.ipynb` | `CROSS` (Фаза 1) |
| `/courses/1-3` | MDP | Value Iteration, уравнения Беллмана | `E01` | `E01_GridWorld` | `E01_GridWorld__value_iteration.ipynb` | `TODO` |
| `/courses/1-4` | Q-Learning: табличный метод | Табличный Q-learning, ε-greedy | `E01` | `E01_GridWorld` | `E01_GridWorld__q_learning.ipynb` | `TODO` |
| `/courses/1-5` | CartPole — первый агент | Дискретизация + Q-learning, PyTorch-агент | `E02` | `E02_CartPoleUnity` | `E02_CartPoleUnity__q_learning.ipynb` | `TODO` |
| `/courses/1-6` | DQN с нуля на PyTorch | **DQN** (replay, target net, Huber) | `E03` | `E03_RollerBall` | `E03_RollerBall__dqn.ipynb` | `TODO` |
| `/courses/1-7` | Exploration vs Exploitation | Бандиты: ε-greedy, **UCB**, **Thompson** | `E00` | `E00_Bandit` | `E00_Bandit__bandits.ipynb` | `TODO` |
| `/courses/project-1` | Проект-1: «Баланс в 3D» | Непрерывное управление, полный цикл | `E04` | `E04_BallBalance` | `E04_BallBalance__ppo.ipynb` | `TODO` |
| `/courses/2-1` | Policy Gradient | **REINFORCE**, baseline | `E05` | `E05_FoodCollector` | `E05_FoodCollector__reinforce.ipynb` | `TODO` |
| `/courses/2-2` | PPO с нуля | **PPO**, clipped objective, **GAE** | `E06` | `E06_Hunter3D` | `E06_Hunter3D__ppo.ipynb` | `TODO` |
| `/courses/2-3` | Непрерывные действия и Actor-Critic | **A2C**, гауссова политика | `E04` | `E04_BallBalance` | `E04_BallBalance__a2c.ipynb` | `TODO` |
| `/courses/2-4` | Reward Shaping | **PBRS**, sparse/dense, ловушки | `E06` | `E06_Hunter3D` | (раздел в ноутбуке E06) | `TODO` |
| `/courses/2-5` | Параллельные среды | Векторизация N арен | — | все среды | — | `CROSS` (`labrl.envs.vec_unity_env`) |
| `/courses/2-6` | TensorBoard и W&B | Схема метрик | — | все среды | — | `CROSS` (`labrl.logging`, раздел 11) |
| `/courses/project-2` | Проект: 3D-охотник | PPO + RayPerception + PBRS | `E06` | `E06_Hunter3D` | `E06_Hunter3D__ppo.ipynb` | `TODO` |
| `/courses/project-3` | Проект: Гоночный агент | SAC, сенсоры, награды | `E07` | `E07_RacingCar` | `E07_RacingCar__sac.ipynb` | `TODO` |
| `/courses/3-1` | SAC — Soft Actor-Critic | **SAC** (max-entropy, 2 Q-критика, авто-α) | `E07` | `E07_RacingCar` | `E07_RacingCar__sac.ipynb` | `TODO` |
| `/courses/3-2` | MA-POCA и Self-Play | **MA-POCA**, CTDE, **Self-Play**, ELO | `E08` | `E08_SoccerArena` | `E08_SoccerArena__mapoca.ipynb` | `TODO` |
| `/courses/3-3` | Учебный план и рандомизация | **Curriculum**, **Domain Randomization**, ADR, PLR | `E09` | `E09_CurriculumMaze` | `E09_CurriculumMaze__ppo_curriculum.ipynb` | `TODO` |
| `/courses/3-4` | Imitation Learning | **BC**, **GAIL** | `E10` | `E10_Imitation` | `E10_Imitation__bc_gail.ipynb` | `TODO` |
| `/courses/3-5` | Деплой: ONNX в Unity | Контракт ONNX, Inference Engine | — | все среды | — | `CROSS` (`labrl.export`, `docs/04_ONNX_CONTRACT.md`) |
| `/courses/3-6` | Оптимизация гиперпараметров | HPO, TPE, Optuna, прунинг | `E07` | `E07_RacingCar` | `E07_RacingCar__hpo.ipynb` | `TODO` |
| `/courses/3-7` | Архитектуры нейросетей | MLP / CNN / LSTM / внимание | — | `E05`, `E07` | (раздел «Нейросеть» каждого ноутбука) | `CROSS` (`labrl.nets`) |
| `/courses/3-8` | Финальный проект | Сквозной пайплайн | — | итог всех | — | `CROSS` (`docs/RESULTS.md`, Фаза 4) |

---

## 2. Реестр примеров (предлагаемый)

| `E##` | Среда | Тип действий | Наблюдения | Алгоритмы | Основание |
|---|---|---|---|---|---|
| `E00` | `E00_Bandit` | Discrete 1×K | вектор (контекст либо пусто) | ε-greedy, UCB, Thompson | урок 1.7, `/labs` |
| `E01` | `E01_GridWorld` | Discrete 1×4 | one-hot 25 | Value Iteration, табличный Q-learning | уроки 1.3, 1.4; **существует** (`Greed_world`, TS-001) |
| `E02` | `E02_CartPoleUnity` | Discrete 1×2 | вектор 4 | Q-learning с дискретизацией, DQN | урок 1.5 (перенос Gymnasium-примера в Unity) |
| `E03` | `E03_RollerBall` | Discrete 1×4 **и** Continuous 2 | вектор 8 | **DQN** (ключевой пример проверки ONNX→Unity) | урок 1.6; **существует** (`Hit_the_ball`) |
| `E04` | `E04_BallBalance` | Continuous 2 | вектор 8 | A2C, PPO | проект-1, урок 2.3, `/unity-projects/ball-balance` |
| `E05` | `E05_FoodCollector` | гибридные | GridSensor + вектор | **REINFORCE** (+ baseline) | урок 2.1, `/unity-projects/food-collector` (flagship курса) |
| `E06` | `E06_Hunter3D` | Continuous 2–3 | RayPerception + вектор | **PPO** + GAE + PBRS | уроки 2.2, 2.4, проект-2 |
| `E07` | `E07_RacingCar` | Continuous 2 | RayPerception | **SAC**, HPO | проект-3, уроки 3.1, 3.6, `/unity-projects/racing` |
| `E08` | `E08_SoccerArena` | смешанные | вектор + Ray | **MA-POCA**, Self-Play | урок 3.2, `/unity-projects/soccer` |
| `E09` | `E09_CurriculumMaze` | Discrete | вектор + Ray | PPO + Curriculum + Domain Randomization | урок 3.3 |
| `E10` | `E10_Imitation` | Discrete/Continuous | вектор | **BC**, **GAIL** | урок 3.4 |
| `E11` | `E11_Research` | по решению автора | по решению автора | авторская архитектура (в т.ч. FCA-based) | `/fca-rl`, раздел 13.3 инструкции, слот `E14_Research` |

---

## 3. Расхождения, требующие решения пользователя

### 3.1. Уроки без примера (`MISSING_EXAMPLE`, если решение — не делать)

`/courses/1-1`, `/courses/1-2` — теоретические/установочные; примера по определению
1.2 инструкции иметь не могут. Предлагается статус `CROSS`, а не `MISSING_EXAMPLE`.

`/courses/2-5`, `/courses/2-6`, `/courses/3-5`, `/courses/3-7`, `/courses/3-8` —
сквозные: реализуются как свойства платформы (векторизация, логирование, экспорт,
архитектуры, итоговая сводка), а не как отдельные среды. Предлагается `CROSS`.

### 3.2. Методы таксономии 13.3 инструкции, которых **нет** в уроках (`NO_LESSON`)

Проверено поиском по всем страницам `cyber-unity-learn` (см. `docs/00_AUDIT.md`, §2.5):

| Метод из 13.3 | Наличие в уроках | Предложение |
|---|---|---|
| Policy Iteration | только вскользь в 3.2 | включить в `E01` вместе с Value Iteration |
| MC, SARSA, Expected SARSA, n-step, Dyna-Q | **отсутствуют** | не делать до решения пользователя |
| `E02_WindyGrid` | **отсутствует** | не делать |
| Double DQN, Dueling DQN | только модуль `/algorithms/dqn`, не урок курса | включить как варианты в `E03` |
| PER, N-step DQN, C51, QR-DQN | **отсутствуют** | не делать |
| DDPG, TD3 | DDPG отсутствует; TD3 — упоминание в 3.1 | не делать (SAC покрывает непрерывное управление) |
| TRPO | упоминание в 2.2 | не делать |
| ICM, RND, sparse reward (`E10_SparseMaze`) | ICM/RND упомянуты в 2.4 | не делать отдельную среду; sparse-режим — опция `E09` |
| A3C | отдельный модуль `/algorithms/a3c`, не урок | не делать |

Это расхождение — предмет вопроса **Q2** в `PLAN.md`.

### 3.3. Существующие материалы, не попавшие в карту (`NO_LESSON`)

Не обнаружено: обе существующие среды (`Hit_the_ball` → `E03`, `Greed_world` → `E01`)
имеют соответствующие уроки.

---

## 4. Правило ведения документа

1. Строка переводится в `DONE` только после выполнения **всех** пунктов DoD раздела 14
   инструкции с подтверждением выводом команд.
2. Каждое изменение статуса сопровождается коммитом с кодом примера
   (`feat(E03): …` / `docs(E03): …`).
3. Behavior Name в Unity **обязан** совпадать с `E##_<Name>` (правило 5.3).
   Текущие имена `RollerAgent` и `GridWorldQLearning` подлежат переименованию в Фазе 1.
