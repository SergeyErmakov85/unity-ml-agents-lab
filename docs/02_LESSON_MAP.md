# 02_LESSON_MAP — Карта «урок → пример → алгоритм»

**Статус документа:** `АКТУАЛЬНЫЙ` — карта подтверждена пользователем (гейт Ф0),
статусы обновляются по мере закрытия примеров.
**Дата:** 2026-08-20
**Источник истины:** фактическое содержимое `C:\cyber-unity-learn`
(`src/content/learningMap.ts`, `src/App.tsx`, страницы `src/pages/*.tsx`) — см. `docs/00_AUDIT.md`, раздел 2.
**Базовый URL уроков:** `https://rl-cuber-unity-code.com`

Статусы: `TODO` — пример запланирован, не начат; `WIP` — в работе;
**`READY`** — сцена, билд, конфиг и ноутбук готовы и проверены, обучение
не запускалось; `DONE` — все пункты DoD (раздел 14 инструкции) выполнены
и подтверждены измерением; `MISSING_EXAMPLE` — урок есть, примера нет
и не запланирован; `NO_LESSON` — пример есть, урока нет; `CROSS` — урок
сквозной, реализуется как свойство всех примеров, а не как отдельный пример.

Разница между `READY` и `DONE` — ровно в запуске обучения. У `READY`-примера
выполнено всё, что не требует полных прогонов: `SceneValidator` без замечаний,
headless-сборка с кодом возврата 0, проверенная связка Python ↔ Unity, экспорт
ONNX по контракту, смоук-тест конвейера. Не выполнено: обучение на трёх сидах,
строка в `docs/RESULTS.md` и проверка инференса по требованию 10.6.

---

## 1. Основная карта

| Урок (маршрут) | Название | Метод | Пример `E##` | Среда | Ноутбук | Статус |
|---|---|---|---|---|---|---|
| `/courses/1-1` | Что такое RL? | Понятия MDP, exploration/exploitation | — | — | — | `CROSS` (теория) |
| `/courses/1-2` | Установка окружения | Стек PyTorch + ML-Agents | — | — | `00_setup_check.ipynb` | `CROSS` (Фаза 1) |
| `/courses/1-3` | MDP | Value Iteration, уравнения Беллмана | `E01` | `E01_GridWorld` | `E01_GridWorld__qlearning.ipynb` (раздел «эталон») | **`DONE`** |
| `/courses/1-4` | Q-Learning: табличный метод | Табличный Q-learning, ε-greedy | `E01` | `E01_GridWorld` | `E01_GridWorld__qlearning.ipynb` | **`DONE`** |
| `/courses/1-5` | CartPole — первый агент | Дискретизация + Q-learning | `E02` | `E02_CartPoleUnity` | `E02_CartPoleUnity__qlearning.ipynb` | **`DONE`** |
| `/courses/1-6` | DQN с нуля на PyTorch | **DQN** (replay, target net, Huber) | `E03` | `E03_RollerBall` | `E03_RollerBall__dqn.ipynb` | **`DONE`** |
| `/courses/1-7` | Exploration vs Exploitation | Бандиты: ε-greedy, **UCB**, **Thompson** | `E00` | `E00_Bandit` | `E00_Bandit__bandits.ipynb` | **`DONE`** |
| `/courses/project-1` | Проект-1: «Баланс в 3D» | Непрерывное управление, полный цикл | `E04` | `E04_BallBalance` | `E04_BallBalance__ppo.ipynb` | **`DONE`** |
| `/courses/2-1` | Policy Gradient | **REINFORCE**, baseline | `E05` | `E05_FoodCollector` | `E05_FoodCollector__reinforce.ipynb` | **`DONE`** |
| `/courses/2-2` | PPO с нуля | **PPO**, clipped objective, **GAE** | `E06` | `E06_Hunter3D` | `E06_Hunter3D__ppo.ipynb` | **`DONE`** |
| `/courses/2-3` | Непрерывные действия и Actor-Critic | **A2C**, гауссова политика | `E04` | `E04_BallBalance` | `E04_BallBalance__a2c.ipynb` | **`DONE`** |
| `/courses/2-4` | Reward Shaping | **PBRS**, sparse/dense, ловушки | `E06`, `E08` | `E06_Hunter3D`, `E08_SoccerArena` | (разделы в ноутбуках) | **`DONE`** |
| `/courses/2-5` | Параллельные среды | Векторизация N арен | — | все среды | — | `CROSS` (`labrl.envs.vec_unity_env`) |
| `/courses/2-6` | TensorBoard и W&B | Схема метрик | — | все среды | — | `CROSS` (`labrl.logging`, раздел 11) |
| `/courses/project-2` | Проект: 3D-охотник | PPO + RayPerception + PBRS | `E06` | `E06_Hunter3D` | `E06_Hunter3D__ppo.ipynb` | **`DONE`** |
| `/courses/project-3` | Проект: Гоночный агент | SAC, сенсоры, награды | `E07` | `E07_RacingCar` | `E07_RacingCar__sac.ipynb` | **`DONE`** |
| `/courses/3-1` | SAC — Soft Actor-Critic | **SAC** (max-entropy, 2 Q-критика, авто-α) | `E07` | `E07_RacingCar` | `E07_RacingCar__sac.ipynb` | **`DONE`** |
| `/courses/3-2` | MA-POCA и Self-Play | **MA-POCA**, CTDE, **Self-Play**, ELO | `E08` | `E08_SoccerArena` | `E08_SoccerArena__mapoca.ipynb` | **`READY`** |
| `/courses/3-3` | Учебный план и рандомизация | **Curriculum**, **Domain Randomization** (ADR и PLR — нет, см. §3.2) | `E09` | `E09_CurriculumMaze` | `E09_CurriculumMaze__ppo_curriculum.ipynb` | **`READY`** |
| `/courses/3-4` | Imitation Learning | **BC**, **GAIL** | `E10` | `E10_Imitation` | `E10_Imitation__bc_gail.ipynb` | **`READY`** |
| `/courses/3-5` | Деплой: ONNX в Unity | Контракт ONNX, Inference Engine | — | все среды | — | `CROSS` (`labrl.export`, `docs/04_ONNX_CONTRACT.md`) |
| `/courses/3-6` | Оптимизация гиперпараметров | HPO: случайный поиск и последовательное деление пополам | `E07` | `E07_RacingCar` | `E07_RacingCar__hpo.ipynb` | **`READY`** |
| `/courses/3-7` | Архитектуры нейросетей | MLP / CNN / LSTM / внимание | — | `E05`, `E07` | (раздел «Нейросеть» каждого ноутбука) | `CROSS` (`labrl.nets`) |
| `/courses/3-8` | Финальный проект | Сквозной пайплайн | — | итог всех | — | `CROSS` (`docs/RESULTS.md`, Фаза 4) |
| `/fca-rl` (модуль) | FCA + RL | Формальный анализ понятий как признаки состояния | `E11` | `E11_Research` | `E11_Research__fca_ppo.ipynb` | **`READY`** |

---

## 2. Реестр примеров

**Статусы обновлены 2026-08-20 по итогам батчей Б2–Б4 Фазы 3.**

| `E##` | Статус | Чем подтверждён |
|---|---|---|
| `E00_Bandit` | **DONE** | три стратегии × 3 сида, IQM 0.7950; сожаление 0.0862 / 0.0074 / 0.0011; ONNX 13/13; инференс в Unity 20/20 `Optimal`, отношение 0.927. Карточка: `docs/envs/E00_Bandit.md` |
| `E01_GridWorld` | **DONE** | 3 сида × награда 0.6800 = точный оптимум; ONNX 13/13; инференс в Unity 20/20 `Goal`, отношение 1.000. Карточка: `docs/envs/E01_GridWorld.md` |
| `E02_CartPoleUnity` | **DONE** | Q-learning с дискретизацией, 3 сида, IQM 199.68 при пороге 195; ONNX 13/13 (сетка внутри графа); инференс в Unity 20/20 по 200 шагов, отношение 1.000. Карточка: `docs/envs/E02_CartPoleUnity.md` |
| `E03_RollerBall` | **DONE** | DQN, успех 100 %; ONNX 13/13; инференс в Unity 20/20 `Goal`, отношение 1.004. Карточка: `docs/envs/E03_RollerBall.md` |
| `E04_BallBalance` | **DONE** | PPO — IQM 100.00 (максимум среды), A2C — 97.12, по 3 сида при пороге 80; ONNX 13/13 для обоих; инференс в Unity 20/20 эпизодов до конца, отношение 1.000. Карточка: `docs/envs/E04_BallBalance.md` |
| `E05_FoodCollector` | **DONE** | REINFORCE + baseline, гибридные действия, `GridSensor`; ONNX 13/13; инференс в Unity проверен. Карточка: `docs/envs/E05_FoodCollector.md` |
| `E06_Hunter3D` | **DONE** | PPO + GAE + потенциальное формирование награды; ONNX 13/13; инференс в Unity проверен. Карточка: `docs/envs/E06_Hunter3D.md` |
| `E07_RacingCar` | **DONE** | SAC на непрерывном управлении; ONNX 13/13; инференс в Unity проверен. Карточка: `docs/envs/E07_RacingCar.md` |
| `E08_SoccerArena` | **READY** | `SceneValidator` без замечаний; билд с кодом возврата 0; Python видит два поведения, 4 группы, obs_dim 64, 178 шаг/с; завершение по времени — 8 матчей по 600 решений, все `truncated`; ONNX 15/15. Карточка: `docs/envs/E08_SoccerArena.md` |
| `E09_CurriculumMaze` | **READY** | `SceneValidator` без замечаний; билд 0; `difficulty` 0/0.33/0.67/1 → сетка 5/7/9/11 на живом билде; базовая линия случайной политики 70/32/7/5 %. Карточка: `docs/envs/E09_CurriculumMaze.md` |
| `E10_Imitation` | **READY** | `SceneValidator` без замечаний; билд 0; случайная политика 0 из 104, эксперт 104 из 104 ровно по 48 ходов; смоук `bc` — 100 % за 18 с, смоук `gail` — 15 000 шагов за 137 с; ONNX 13/13. Карточка: `docs/envs/E10_Imitation.md` |
| `E11_Research` | **READY** | `SceneValidator` без замечаний; билд 0; решётка из 19 понятий, эквивалентность «есть ключ ≡ дверь открыта» обнаружена методом самостоятельно; смоук `fca_ppo` и контрольного `ppo_discrete` — оба 100 %; ONNX 13/13. Карточка: `docs/envs/E11_Research.md` |

### Полный реестр

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

Это расхождение — предмет вопроса **Q2** в `PLAN.md`. Решение пользователя:
делать 12 примеров `E00`–`E11` строго по фактическим урокам; перечисленное
выше помечено `NO_LESSON` и не реализуется.

**Что из этого списка всё же появилось по ходу работы, и почему:**

| Метод | Где появился | Почему |
|---|---|---|
| Double DQN | `E03_RollerBall`, ключ `double_dqn` | вариант того же алгоритма, а не отдельный метод |
| Domain Randomization | `E09_CurriculumMaze` | неотделим от curriculum: без него агент выучивает планировку, а не умение |
| PBRS | `E06_Hunter3D`, `E08_SoccerArena` | урок 2.4 его требует, и без него `E08` необучаема (измерено) |

**Что из урока 3.3 сознательно НЕ реализовано:** ADR (Automatic Domain
Randomization) и PLR (Prioritized Level Replay). Оба требуют отдельной
инфраструктуры — автоматического расширения границ распределения по успеху
и хранилища уровней с приоритетом по обучающему сигналу. Зафиксировано
в `ENV_SPEC.md` среды `E09`, §2 и в `docs/algos/curriculum.md`.

**Что из урока 3.4 сознательно НЕ реализовано:** DAgger. Требует эксперта
**в цикле** обучения, а не набора записей, — то есть другой архитектуры
взаимодействия (`ENV_SPEC.md` среды `E10`, §2).

**Что из урока 3.6 реализовано иначе, чем в уроке:** урок использует Optuna
и W&B Sweeps. Здесь HPO реализован своим кодом — случайный поиск
и последовательное деление пополам (successive halving), — потому что
запрет 16.5 распространяется на учебные реализации, а тянуть внешнюю
зависимость ради двух алгоритмов подбора несоразмерно. Optuna при этом
не запрещена: она не RL-библиотека, и подключить её к тому же интерфейсу
пользователь может сам.

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
   Переименование выполнено в Фазе 1; `SceneValidator` проверяет это на каждой сборке.
