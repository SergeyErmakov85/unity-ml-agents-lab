# TS-001. GridWorld — среда для Q-learning

<!-- validator: obs_size=25; discrete_branches=4; continuous_size=0; max_step=100 -->

Строка выше — машиночитаемый контракт для `SceneValidator` (формат:
`docs/03_CONVENTIONS.md`, §3). Значения в ней обязаны совпадать с разделами
7.2–7.3 этого документа и с фактической сценой; расхождение — ошибка сборки.

## 0. Метаданные

| Параметр | Значение |
|---|---|
| Идентификатор среды | `E01_GridWorld` (он же **Behavior Name**, правило 5.3) |
| Unity | 6000.5.4f1 |
| ML-Agents | `com.unity.ml-agents` 4.0.3; Python `mlagents` 1.2.0.dev0 (`release_23_tag`) |
| Render Pipeline | URP 17.5.0 |
| Версия TS | 1.1 |
| Дата | 2026-08-15 (ред. 1.1: приведение к стандарту 7.2) |
| Целевой исполнитель | Claude Code |

### Изменения в редакции 1.1

Среда приведена к стандарту раздела 7 инструкции проекта:

* `GridWorldEnvironment` наследует `LabRL.Core.TrainingAreaBase` — сид и
  сложность приходят из Python через `EnvironmentParametersChannel`, у каждой
  арены **свой** генератор случайных чисел;
* `GridWorldAgent` наследует `LabRL.Core.AgentBase` — сверка Behavior Name
  с идентификатором среды и публикация метрик в неймспейс `Env/`;
* в сцене **4 арены** из одного префаба с шагом 8 по X: Python трактует их как
  4 параллельные среды (требование 8.2);
* корневые группы сцены: `TrainingAreas`, `Cameras`, `Lighting`, `Managers`,
  `UI`, `Debug`;
* Behavior Name изменён с `GridWorldQLearning` на `E01_GridWorld`.

### Изменения от 2026-09-14

* Камера (раздел 8) показывает все 4 арены, а не только арену 0.

---

## 1. Краткое описание и цель

Среда реализует классический дискретный **Grid World** размером 5×5 клеток для обучения с подкреплением табличным методом Q-learning. Агент стартует в левой нижней клетке (0,0) и должен достичь целевой клетки (4,4), избегая двух клеток-ловушек и обходя три непроходимые стены. Переходы детерминированы, состояние дискретно (индекс клетки 0–24), действия дискретны (четыре направления). Критерий успеха обучения: агент стабильно находит один из кратчайших маршрутов до цели, не заходя в ловушки. Среда пригодна одновременно для табличного Q-learning (через открытый индекс состояния) и для нейросетевого обучения ML-Agents (через one-hot наблюдение).

---

## 2. Архитектура сцены

- Масштаб: **1 unit = 1 клетка = 1 м**.
- Пространство арены: **5×5 юнитов** по осям X (столбцы) и Z (ряды), высота стен 1 юнит по Y.
- Система координат: центр арены — клетка (2,2) — совпадает с началом мира (0,0,0).
- Формула перевода клетки в мир: `world(c, r) = ((c − 2)·1.0,  y,  (r − 2)·1.0)`, где `c` — столбец (ось X), `r` — ряд (ось Z), оба ∈ {0…4}.
- Границы поверхности: x ∈ [−2.5; 2.5], z ∈ [−2.5; 2.5].
- Индекс состояния: `state = r·5 + c`, диапазон [0; 24].
- Количество TrainingArea: **1** (Prefab `TrainingArea`; для параллельного обучения инстанцируется K раз со смещением 8 юнитов по оси X — см. раздел 12).

### Карта среды (r — ряд сверху вниз, c — столбец слева направо)

```
r=4:  .    .    .    W    G
r=3:  .    T    .    .    .
r=2:  .    W    .    .    .
r=1:  .    .    .    T    .
r=0:  S    .    .    .    .

S — старт (0,0)   G — цель (4,4)
T — ловушка (1,3),(3,1)   W — стена (1,2),(3,3),(3,4)
. — свободная клетка
```

Проверенный кратчайший маршрут (8 ходов): (0,0)→(1,0)→(2,0)→(3,0)→(4,0)→(4,1)→(4,2)→(4,3)→(4,4). Маршрут не пересекает стены и ловушки, что подтверждает разрешимость карты.

---

## 3. Инвентарь объектов

| Имя | Тип | Роль в RL | Кол-во | Prefab |
|---|---|---|---|---|
| `TrainingArea` | Empty (контейнер + контроллер) | инфраструктура | 1 | Да |
| `Ground` | Cube | decoration | 1 | Нет |
| `GridLines` | Quad | decoration | 1 | Нет |
| `StartMarker` | Cube (плоский) | decoration | 1 | Нет |
| `GoalMarker` | Cube (плоский) | target | 1 | Нет |
| `TrapMarker` | Cube (плоский) | reward-zone (−) | 2 | Да |
| `Wall` | Cube | obstacle | 3 | Да |
| `Agent` | Sphere | agent | 1 | Нет |
| `UICanvas` | Canvas + TextMeshPro | инфраструктура (UI) | 1 | Нет |
| `MainCamera` | Camera | инфраструктура | 1 | Нет |
| `DirectionalLight` | Light | инфраструктура | 1 | Нет |

---

## 4. Иерархия

```
GridWorld (Scene)
├── TrainingArea_01                 [GridWorldEnvironment.cs]
│   ├── Ground
│   ├── GridLines
│   ├── Markers
│   │   ├── StartMarker
│   │   ├── GoalMarker
│   │   ├── TrapMarker_01
│   │   └── TrapMarker_02
│   ├── Walls
│   │   ├── Wall_01
│   │   ├── Wall_02
│   │   └── Wall_03
│   └── Agent                       [BehaviorParameters, DecisionRequester, GridWorldAgent.cs]
├── UICanvas                        [GridWorldUI.cs]
│   └── StatsPanel
│       ├── Text_Episode
│       ├── Text_Step
│       ├── Text_State
│       ├── Text_LastAction
│       ├── Text_Reward
│       └── Text_Result
├── MainCamera
└── DirectionalLight
```

---

## 5. Параметры объектов

### 5.1. `TrainingArea_01`

| Поле | Значение |
|---|---|
| Тип | Empty GameObject |
| Роль в RL | контейнер среды + контроллер |
| Transform | Position (0, 0, 0); Rotation (0, 0, 0); Scale (1, 1, 1) |
| Collider | нет |
| Rigidbody | нет |
| Скрипты | `GridWorldEnvironment` (серализованные поля — см. раздел 6.1) |

### 5.2. `Ground`

| Поле | Значение |
|---|---|
| Роль в RL | decoration |
| Геометрия | Cube |
| Transform | Position (0, −0.05, 0); Rotation (0,0,0); Scale (5, 0.1, 5) |
| Материал | `Mat_Ground`, цвет #2B2B33, Metallic 0.0, Smoothness 0.1 |
| Collider | BoxCollider (авто от Cube), isTrigger = false |
| Rigidbody | нет |
| Взаимодействия | нет (логика клеточная, индексная) |

Верхняя грань поверхности находится на y = 0.0.

### 5.3. `GridLines`

| Поле | Значение |
|---|---|
| Роль в RL | decoration |
| Геометрия | Quad |
| Transform | Position (0, 0.011, 0); Rotation (90, 0, 0); Scale (5, 5, 1) |
| Материал | `Mat_GridLines` (Unlit, Transparent), линии #4A4A57, шаг сетки 1 юнит, толщина линии 0.03 юнита |
| Collider | нет |
| Rigidbody | нет |

### 5.4. `StartMarker`

| Поле | Значение |
|---|---|
| Роль в RL | decoration (визуальная отметка старта) |
| Геометрия | Cube |
| Transform | Position (−2, 0.011, −2); Rotation (0,0,0); Scale (0.9, 0.02, 0.9) |
| Материал | `Mat_Start`, цвет #F1C40F |
| Collider | нет |
| Rigidbody | нет |

### 5.5. `GoalMarker`

| Поле | Значение |
|---|---|
| Роль в RL | target |
| Геометрия | Cube |
| Transform | Position (2, 0.011, 2); Rotation (0,0,0); Scale (0.9, 0.02, 0.9) |
| Материал | `Mat_Goal`, цвет #2ECC71 |
| Collider | BoxCollider, isTrigger = true (не обязателен для логики; логика — индексная) |
| Rigidbody | нет |
| Тег | `goal` |

### 5.6. `TrapMarker` (Prefab, 2 инстанса)

| Поле | Значение |
|---|---|
| Роль в RL | reward-zone (отрицательная) |
| Геометрия | Cube |
| Transform (общий) | Rotation (0,0,0); Scale (0.9, 0.02, 0.9); Position.y = 0.011 |
| Позиции инстансов | `TrapMarker_01`: (−1, 0.011, 1) — клетка (1,3); `TrapMarker_02`: (1, 0.011, −1) — клетка (3,1) |
| Материал | `Mat_Trap`, цвет #E74C3C |
| Collider | BoxCollider, isTrigger = true (опционально) |
| Rigidbody | нет |
| Тег | `trap` |

### 5.7. `Wall` (Prefab, 3 инстанса)

| Поле | Значение |
|---|---|
| Роль в RL | obstacle (непроходимая клетка) |
| Геометрия | Cube |
| Transform (общий) | Rotation (0,0,0); Scale (1, 1, 1); Position.y = 0.5 |
| Позиции инстансов | `Wall_01`: (−1, 0.5, 0) — клетка (1,2); `Wall_02`: (1, 0.5, 1) — клетка (3,3); `Wall_03`: (1, 0.5, 2) — клетка (3,4) |
| Материал | `Mat_Wall`, цвет #7F8C8D, Metallic 0.0, Smoothness 0.1 |
| Collider | BoxCollider, isTrigger = false |
| Rigidbody | нет |
| Тег | `wall` |

### 5.8. `Agent`

| Поле | Значение |
|---|---|
| Роль в RL | agent |
| Геометрия | Sphere |
| Transform | Position (−2, 0.3, −2) — клетка (0,0); Rotation (0,0,0); Scale (0.6, 0.6, 0.6) |
| Материал | `Mat_Agent`, цвет #3498DB |
| Collider | SphereCollider, isTrigger = false (в логике движения не используется) |
| Rigidbody | нет (перемещение — телепорт через `transform.position`) |
| Скрипты | `BehaviorParameters`, `DecisionRequester`, `GridWorldAgent` (параметры — раздел 6.2, 7) |
| Тег | `agent` |

### 5.9. `UICanvas` / `MainCamera` / `DirectionalLight`

Параметры — разделы 11, 8, 9 соответственно.

### 5.10. Теги и слои

- Требуемые теги (создать в Tag Manager): `agent`, `goal`, `trap`, `wall`.
- Слои: только стандартный `Default`. Пользовательские слои не требуются.
- Замечание: вся игровая логика опирается на индекс клетки, а не на физические столкновения; теги заданы для опциональной триггерной валидации и отладки.

---

## 6. Поведение объектов

Статика (Collider без Rigidbody): `Ground`, `Wall`, все маркеры. Динамика: только `Agent`, перемещаемый скриптом дискретными телепортами между центрами клеток. Физическая симуляция движения не используется — переходы детерминированы и мгновенны.

### 6.1. Скрипт `GridWorldEnvironment` (на `TrainingArea_01`)

Серализованные поля (значения по умолчанию):

| Поле | Тип | Значение |
|---|---|---|
| `cols` | int | 5 |
| `rows` | int | 5 |
| `cellSize` | float | 1.0 |
| `startCell` | Vector2Int | (0, 0) |
| `goalCell` | Vector2Int | (4, 4) |
| `trapCells` | List\<Vector2Int\> | [(1,3), (3,1)] |
| `wallCells` | List\<Vector2Int\> | [(1,2), (3,3), (3,4)] |
| `stepReward` | float | −0.04 |
| `goalReward` | float | +1.0 |
| `trapReward` | float | −1.0 |
| `maxSteps` | int | 100 |
| `randomStart` | bool | false |
| `slipProbability` | float | 0.0 |

Методы (сигнатуры обязательны):

- `Vector3 CellToWorld(Vector2Int cell, float y)` → `((cell.x − 2)·cellSize, y, (cell.y − 2)·cellSize)`.
- `bool InBounds(Vector2Int cell)` → `cell.x ∈ [0; cols−1] && cell.y ∈ [0; rows−1]`.
- `bool IsWall(Vector2Int cell)`, `bool IsTrap(Vector2Int cell)`, `bool IsGoal(Vector2Int cell)`.
- `int StateIndex(Vector2Int cell)` → `cell.y·cols + cell.x`.
- `Vector2Int ResolveMove(Vector2Int cell, int action)` — применяет направление; при выходе за границы или попадании на стену возвращает исходную `cell` (ход блокирован).
  - Направления: `0 = North` (r+1, +Z); `1 = South` (r−1, −Z); `2 = East` (c+1, +X); `3 = West` (c−1, −X).
  - При `slipProbability > 0`: с вероятностью `slipProbability` выбранное действие заменяется случайным из двух перпендикулярных направлений (модель проскальзывания FrozenLake). По умолчанию 0 — детерминизм.
- `Vector2Int RandomFreeCell()` — равномерный выбор клетки из множества `{все клетки} \ ({wallCells} ∪ {goalCell} ∪ {trapCells})`.

### 6.2. Скрипт `GridWorldAgent` (на `Agent`, наследник `Agent` ML-Agents)

- Поле `currentCell : Vector2Int`.
- Публичное свойство `int CurrentStateIndex => env.StateIndex(currentCell)` — предназначено для внешней реализации табличного Q-learning.
- Логика методов описана в разделе 7.

### 6.3. Скрипт `GridWorldUI` (на `UICanvas`)

Каждый шаг читает из `GridWorldEnvironment`/`GridWorldAgent` и обновляет текстовые поля (раздел 11).

---

## 7. Среда Reinforcement Learning

### 7.1. Агент

- Визуальное представление: синяя сфера Ø 0.6 юнита, центр на высоте y = 0.3.
- Компоненты: `BehaviorParameters`, `DecisionRequester`, `GridWorldAgent`.
- `BehaviorParameters`:
  - Behavior Name: `GridWorldQLearning`.
  - Vector Observation → Space Size: **25**; Stacked Vectors: 1.
  - Actions → Discrete, Branches: 1, Branch 0 Size: **4**.
  - Model: отсутствует (обучение с нуля).
- `DecisionRequester`: Decision Period = **1**; Take Actions Between Decisions = false (пошаговый режим, один ход = одно решение).
- `Agent.MaxStep` = **100**.
- Стартовое условие: `currentCell = randomStart ? RandomFreeCell() : startCell`.

### 7.2. Observation Space

| Наблюдение | Источник | Тип | Размерность | Нормализация |
|---|---|---|---|---|
| One-hot индекса состояния | `StateIndex(currentCell)` | float[25] | 25 | значения ∈ {0, 1} (уже нормализованы) |

`CollectObservations(VectorSensor sensor)`: сформировать массив из 25 нулей, установить 1.0 в позиции `StateIndex(currentCell)`, передать в сенсор.

Обоснование выбора: one-hot делает нейросетевую политику эквивалентной табличному представлению Q-значений, что согласует среду с методом Q-learning. Карта фиксирована между эпизодами, поэтому наблюдение только собственной позиции достаточно (значения клеток агент выучивает). Альтернатива — 2-мерное нормализованное `(c/(cols−1), r/(rows−1))` (см. Реестр допущений A-6).

### 7.3. Action Space

- Тип: Discrete, одна ветвь, размер 4.
- Семантика индексов: `0 = North` (+Z), `1 = South` (−Z), `2 = East` (+X), `3 = West` (−X).
- Значение действия соответствует попытке перехода в соседнюю клетку; при блокировке (граница/стена) агент остаётся в текущей клетке.
- `Heuristic(in ActionBuffers)`: стрелка Вверх → 0, Вниз → 1, Вправо → 2, Влево → 3 (ручное управление для проверки).

### 7.4. Reward Design

| Событие | Значение | Частота | Завершает эпизод |
|---|---|---|---|
| Каждый шаг (`OnActionReceived`, применяется до проверки цели/ловушки) | −0.04 | за шаг | Нет |
| Переход в клетку `goalCell` | +1.0 | однократно | Да |
| Переход в клетку из `trapCells` | −1.0 | однократно | Да |
| Заблокированный ход (граница/стена) | 0.0 (доп. штрафа нет; действует только шаговый −0.04) | за шаг | Нет |
| Достижение `MaxStep` = 100 без цели | 0.0 (терминального бонуса/штрафа нет) | однократно | Да (таймаут) |

Итоговая доходность кратчайшего маршрута (8 ходов): `+1.0 − 8·0.04 = +0.68`. Маршрут в 16 ходов: `+0.36`. Заход в ловушку: `≤ −1.0`.

**Проверка на reward hacking.** Шаговый штраф −0.04 исключает стратегию блуждания и стимулирует кратчайший путь. Штраф ловушки −1.0 терминален и по модулю превосходит любую достижимую экономию шаговых штрафов, поэтому агент не использует ловушку как «быстрое» завершение эпизода. Награда цели +1.0 терминальна и положительна — единственный аттрактор; повторное начисление положительной награды невозможно, так как цель завершает эпизод (нет фарма). Заблокированный ход тратит шаг без прогресса, что делает удары в стену невыгодными без отдельного штрафа.

### 7.5. Логика эпизода

`OnEpisodeBegin()`:
1. `currentCell = randomStart ? RandomFreeCell() : startCell`.
2. `transform.position = env.CellToWorld(currentCell, 0.3f)`.
3. Сбросить счётчики шага и накопленной награды (для UI).

`OnActionReceived(ActionBuffers actions)`:
1. `int a = actions.DiscreteActions[0]`.
2. `Vector2Int next = env.ResolveMove(currentCell, a)`.
3. `currentCell = next; transform.position = env.CellToWorld(currentCell, 0.3f)`.
4. `AddReward(stepReward)`.
5. Если `IsGoal(currentCell)`: `AddReward(goalReward); EndEpisode();`.
6. Иначе если `IsTrap(currentCell)`: `AddReward(trapReward); EndEpisode();`.

Завершение эпизода: достижение цели, заход в ловушку либо `MaxStep = 100` (таймаут, обрабатывается ML-Agents). После завершения агент возвращается в стартовую клетку согласно `OnEpisodeBegin`.

---

## 8. Камера

| Параметр | Значение |
|---|---|
| Имя | `MainCamera` |
| Проекция | Orthographic |
| Orthographic Size | 9.6875 = (12 + 2.5 + 1) / 1.6 |
| Position | (12, 10, 0) — центр ряда из 4 арен |
| Rotation | (90, 0, 0) — вид строго сверху вниз |
| Clear Flags | Solid Color, цвет #202028 |
| В кадре | все 4 арены (X от −2.5 до 26.5) с полем 1 юнит по бокам при соотношении сторон 16:10 и шире |

Размер и позиция вычисляются в `GridWorldSetup` из `AreaCount`, `AreaSpacing`,
`AreaSize`, `CameraMargin` и `CameraMinAspect`. На экране уже 16:10 (например, 4:3)
крайние арены обрезаются по бокам.

---

## 9. Освещение

| Параметр | Значение |
|---|---|
| Тип | Directional Light |
| Rotation | (50, −30, 0) |
| Intensity | 1.0 |
| Color | #FFF4E5 |
| Shadows | Soft, Strength 0.6 |
| Ambient Source | Color, #404050, Intensity 1.0 |

---

## 10. Визуальный стиль

| Элемент | HEX | Материал |
|---|---|---|
| Фон камеры / Clear | #202028 | — |
| Поверхность `Ground` | #2B2B33 | `Mat_Ground` (URP Lit) |
| Линии сетки | #4A4A57 | `Mat_GridLines` (URP Unlit, Transparent) |
| Маркер старта | #F1C40F | `Mat_Start` |
| Маркер цели | #2ECC71 | `Mat_Goal` |
| Маркер ловушки | #E74C3C | `Mat_Trap` |
| Стена | #7F8C8D | `Mat_Wall` |
| Агент | #3498DB | `Mat_Agent` |
| Текст UI | #ECF0F1 | TextMeshPro |

Уровень детализации: примитивы Unity, материалы URP Lit (Metallic 0.0, Smoothness 0.1) кроме `Mat_GridLines` (Unlit). Дополнительная геометрия и текстуры не применяются.

---

## 11. UI / Canvas

| Параметр | Значение |
|---|---|
| `UICanvas` | Render Mode = Screen Space – Overlay |
| Canvas Scaler | Scale With Screen Size; Reference Resolution 1920×1080; Match 0.5 |
| `StatsPanel` | Anchor = верхний левый угол; Position (20, −20); ширина 320, высота 200; фон #202028 с Alpha 0.6 |

Текстовые поля (TextMeshProUGUI, цвет #ECF0F1, размер 22), обновляются каждый шаг скриптом `GridWorldUI`:

| Поле | Формат |
|---|---|
| `Text_Episode` | `Episode: {n}` |
| `Text_Step` | `Step: {k} / 100` |
| `Text_State` | `State: {index}` (0–24) |
| `Text_LastAction` | `Action: {N/S/E/W или -}` |
| `Text_Reward` | `Reward: {sum:F2}` |
| `Text_Result` | `Result: {Running / Goal / Trap / Timeout}` |

---

## 12. Инженерные замечания и рекомендации по реализации

1. **Двойной режим обучения.** Среда поддерживает как нейросетевое обучение ML-Agents (PPO/SAC через one-hot наблюдение), так и внешний табличный Q-learning через свойство `CurrentStateIndex` (0–24) и дискретное действие 0–3. Переходы детерминированы (при `slipProbability = 0`), поэтому таблица Q заполняется корректно.
2. **Телепорт вместо физики.** Перемещение агента реализовано установкой `transform.position` в центр клетки; Rigidbody отсутствует. Это гарантирует детерминизм, устраняет накопление физических ошибок и ускоряет обучение.
3. **Индексная логика вместо коллайдеров.** Награды и завершение эпизода вычисляются по индексу клетки, а не по столкновениям. Коллайдеры маркеров цели/ловушек (`isTrigger = true`) добавлены для опциональной триггерной отладки и не влияют на MDP.
4. **Масштабируемость.** `TrainingArea` — Prefab. Для параллельного обучения инстанцировать K копий со смещением 8 юнитов по оси X; каждый экземпляр содержит собственные `GridWorldEnvironment` и `Agent`, координаты клеток задаются локально относительно контейнера.
5. **Параметризация карты.** Размер сетки, позиции старта/цели/ловушек/стен и значения наград вынесены в серализованные поля `GridWorldEnvironment` — изменение карты не требует правок кода.
6. **Опциональная стохастичность.** Поле `slipProbability` эмулирует скользкую среду типа FrozenLake; по умолчанию 0 (детерминизм) для чистой демонстрации сходимости Q-learning.
7. **Out of scope (согласно Playbook 2.3).** Гиперпараметры обучения (learning rate, gamma, epsilon, batch size), выбор и конфигурация алгоритма, инфраструктура тренировки в данную TS не входят и задаются вне сцены.

---

## 13. Реестр допущений

| № | Допущение | Основание | Как изменить |
|---|---|---|---|
| A-1 | Термин «Canvas» из запроса трактуется как Unity Scene (сцена целиком); UI-Canvas используется только в разделе 11 | Глоссарий Playbook (гл. 3) запрещает синонимию «Canvas»↔«сцена» | — |
| A-2 | Сетка 5×5, клетка 1×1 юнит | Канонический размер учебного grid world | Поля `cols`, `rows`, `cellSize` |
| A-3 | Карта: старт (0,0), цель (4,4), ловушки (1,3),(3,1), стены (1,2),(3,3),(3,4) | Разрешимая нетривиальная раскладка с проверенным маршрутом | Поля `startCell`, `goalCell`, `trapCells`, `wallCells` |
| A-4 | 4 дискретных действия (N/S/E/W), без «стоять» | Каноническая формулировка grid world | Увеличить Branch Size до 5 и добавить обработку `stay` |
| A-5 | Фиксированный старт (0,0), детерминированные переходы | Воспроизводимость демонстрации Q-learning | Поля `randomStart`, `slipProbability` |
| A-6 | Наблюдение — one-hot 25 | Эквивалентность табличному представлению | Заменить на 2-мерное нормализованное `(c,r)`, Space Size = 2 |
| A-7 | Награды: шаг −0.04, цель +1.0, ловушка −1.0; MaxStep 100 | Шейпинг кратчайшего пути (стиль Russell–Norvig) | Поля `stepReward`, `goalReward`, `trapReward`, `maxSteps` |
| A-8 | Одна TrainingArea | Наглядность учебной сцены | Инстанцировать Prefab K раз, смещение 8 юнитов по X |
| A-9 | Render Pipeline URP | Актуальный стек примеров ML-Agents | Заменить материалы на Built-in |
| A-10 | Агент — сфера Ø 0.6 | Чёткая различимая фишка | Заменить меш агента |
| A-11 | Линии сетки — текстурный/Unlit Quad | Минимум draw calls | Заменить на LineRenderer или 25 отдельных тайлов |

---

## 14. Критерии приёмки

- [ ] Сцена содержит сетку 5×5; агент, маркер цели, 2 маркера ловушек и 3 стены размещены в указанных клетках.
- [ ] `BehaviorParameters`: Behavior Name = `GridWorldQLearning`, Observation Space Size = 25, одна дискретная ветвь размером 4.
- [ ] `Agent.MaxStep` = 100; `DecisionRequester.Decision Period` = 1.
- [ ] Переход в клетку (4,4) даёт +1.0 и завершает эпизод.
- [ ] Переход в клетку (1,3) или (3,1) даёт −1.0 и завершает эпизод.
- [ ] Каждый шаг даёт −0.04; заблокированный ход оставляет агента в текущей клетке.
- [ ] Агент не проходит сквозь стены и не покидает границы 5×5.
- [ ] Камера ортографическая, вид сверху, в кадре все 4 арены с полем 1 юнит (16:10 и шире).
- [ ] UI-Canvas отображает episode, step, state index, last action, cumulative reward, result.
- [ ] Свойство `CurrentStateIndex` возвращает `r·5 + c` ∈ [0; 24].
- [ ] В режиме Heuristic нажатие стрелки перемещает агента на одну клетку за нажатие с учётом стен и границ.
- [ ] После цели/ловушки/таймаута эпизод автоматически перезапускается, агент возвращается в клетку (0,0).
- [ ] `TrainingArea` является Prefab; его дублирование создаёт независимую рабочую область.

---

*Самопроверка по Playbook v3.0 пройдена: 20/20 пунктов.*
