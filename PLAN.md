# PLAN — План работ по превращению `unity-ml-agents-lab` в учебно-исследовательскую платформу

**Дата:** 2026-08-15
**Ветка:** `feature/lab-bootstrap`
**Основание:** `CLAUDE_Unity-ml-agents-lab.md`
**Текущая фаза:** Фаза 2 (вертикальный срез) — выполняется. Состояние — раздел 7.

Сопутствующие документы: `docs/00_AUDIT.md`, `docs/01_STACK.md`,
`docs/02_LESSON_MAP.md`, `docs/04_ONNX_CONTRACT.md`, `docs/ASSUMPTIONS.md`.

---

## 1. Что сделано в Фазе 0

| Пункт Фазы 0 | Статус | Артефакт |
|---|---|---|
| 1. Обход обоих репозиториев | ✔ выполнено | `docs/00_AUDIT.md` §1 |
| 2. Состав уроков `cyber-unity-learn` и перечень методов RL | ✔ выполнено | `docs/00_AUDIT.md` §2 |
| 3. Инвентаризация лаборатории | ✔ выполнено | `docs/00_AUDIT.md` §3 |
| 4. Верификация стека с выводами команд | ✔ выполнено, обнаружены блокеры B-1/B-2 | `docs/01_STACK.md` |
| 5. Извлечение контракта ONNX | ⚠ выполнено из исходников `release_23_tag`, **не** из установленного пакета (B-2) | `docs/04_ONNX_CONTRACT.md` (статус `DRAFT`) |
| 6. Файлы `00_AUDIT`, `01_STACK`, `02_LESSON_MAP`, `04_ONNX_CONTRACT`, `PLAN.md` | ✔ созданы (+ `ASSUMPTIONS.md`) | этот каталог |
| 7. Не более 5 вопросов и остановка | ✔ раздел 4 этого документа | — |

Ничего не устанавливалось, не переносилось и не удалялось.
В `C:\cyber-unity-learn` не выполнено ни одной операции записи.

---

## 2. Ключевые находки, определяющие план

1. **Unity 6000.5.4f1 не установлен.** Установлены 2021.3.45f1, 2023.2.13f1,
   2023.2.20f1, 6000.0.35f1, 6000.3.10f1, 6000.5.0a8. Ни одна проверка на стороне
   Unity (компиляция, `SceneValidator`, headless-сборка, инференс ONNX) сейчас
   невозможна.
2. **Python-стек RL отсутствует полностью**, а единственный рабочий интерпретатор —
   3.11.4, тогда как `mlagents`/`mlagents_envs` требуют `>=3.10.1,<=3.10.12`.
3. **Нужная версия `mlagents` не публикуется на PyPI** (там максимум 0.28.0);
   ставится только из git — как и учит урок 1.2 курса.
4. **Контракт ONNX полностью восстановлен** и не содержит неизвестных: opset 9,
   входы `obs_i` / `action_masks` / `recurrent_in`, выходы `version_number`(=3),
   `memory_size`, `{continuous,discrete}_actions`, `*_output_shape`,
   `deterministic_*`, при рекуррентности `recurrent_out`.
5. **Фактический курс уже, чем таксономия 13.3 инструкции.** В уроках нет SARSA,
   MC, n-step, Dyna-Q, DDPG, TD3, PER, Rainbow, C51/QR-DQN, A3C как урока.
6. **Целевая структура раздела 5 инструкции противоречит последнему коммиту репозитория.**
   Инструкция требует `unity/MLAgentsLab/…`, а коммит `c35e558` намеренно сделал
   корень репозитория Unity-проектом (и это зафиксировано в `CLAUDE.md`).
7. **GPU есть:** NVIDIA GeForce GTX 1650, 4 ГБ, драйвер 560.94 — PyTorch с CUDA 12.1
   поддерживается; 4 ГБ достаточно для MLP/CNN-политик учебного масштаба.

---

## 3. План по фазам

### Фаза 1 — Каркас платформы

**Предусловие:** сняты блокеры B-1 (Unity) и B-2 (Python) — см. вопросы Q1.

1. Коммит-снимок `chore(repo): snapshot before restructure` (требование 3.5).
2. **Реструктуризация** по разделу 5 через `git mv`, вытесненное — в
   `_archive/2026-08-15/` с сохранением относительных путей:
   - `Assets/ML-ENVIRONMENTS/02-Examples/Greed_world/` → `…/Assets/Envs/E01_GridWorld/`
     (`INSTRUCTIONS.md` → `ENV_SPEC.md`);
   - `Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/` → `…/Assets/Envs/E03_RollerBall/`;
   - `Assets/Editor/ProjectBootstrap.cs` → `…/Assets/Shared/Editor/`;
   - пустые категории `03-Classic-Games`…`10-Research` — удалить как каталоги-заглушки
     (только `.gitkeep`, содержимого нет; фиксируется в `00_AUDIT.md`).
   - Behavior Names: `GridWorldQLearning` → `E01_GridWorld`, `RollerAgent` → `E03_RollerBall`.
3. **Python-окружение:** `python/.venv` на Python 3.10.x, `python/pyproject.toml`
   с границами версий, `python/requirements.lock.txt` (`pip freeze`), вывод `pip check`
   в `docs/01_STACK.md`.
4. **Пересверка контракта ONNX** с установленным пакетом → снятие статуса `DRAFT`.
5. **Пакет `labrl`:** `envs/` (`unity_env`, `vec_unity_env`, `side_channels`, `registry`),
   `nets/protocols.py` + `mlp.py`, `logging/` (`tb_logger`, `run_dir`),
   `utils/` (`seeding`, `schedules`, `config`, `checkpoint`),
   `export/` (`onnx_export`, `onnx_verify`) — критический модуль,
   `eval/` (`evaluate`, `aggregate`).
6. **Unity Shared:** `Assets/Shared/Scripts/Core` (`AgentBase`, `TrainingAreaBase`,
   `SpawnService`, `MetricsRecorder`), `Assets/Shared/Editor/BuildScript.cs`
   (headless-сборка, ненулевой код возврата при ошибке), `SceneValidator.cs`,
   `OnnxContractCheck.cs`.
7. **Шаблоны:** `docs/templates/{TS_TEMPLATE,ENV_CARD,ALGO_CARD}.md`,
   `notebooks/00_setup_check.ipynb`.
8. **Тесты:** `python/tests/` — контракт ONNX, формы тензоров, корректность
   `terminated` vs `truncated` (обязательный тест по 8.3), smoke-обучение.
9. **Документы:** `README.md`, `CLAUDE.md` (обновление), `docs/03_CONVENTIONS.md`,
   `docs/06_WORKFLOW.md`, `docs/05_TENSORBOARD.md`, `docs/adr/ADR-0001-single-unity-project.md`.

**Гейт Ф1:** `00_setup_check.ipynb` подключается и к Editor, и к headless-билду,
печатает размерности пространств. Остановка.

### Фаза 2 — Вертикальный срез

1. `E01_GridWorld` — табличный Q-learning + Value Iteration (внешний Python,
   через `CurrentStateIndex`; проверка MDP и связки Python↔Unity).
2. `E03_RollerBall` — **DQN на PyTorch** (дискретный вариант действий уже
   предусмотрен в коде агента): полный путь сцена → билд → ноутбук → обучение →
   TensorBoard → ONNX → верификация → инференс в Unity на 20 эпизодах.
3. Все грабли — в `docs/07_TROUBLESHOOTING.md`.

**Гейт Ф2:** пользователь лично видит в Unity агента под управлением ONNX-модели
и графики в TensorBoard. Остановка.

### Фаза 3 — Тиражирование батчами по 3 примера

| Батч | Примеры | Основные методы |
|---|---|---|
| Б1 | `E00_Bandit`, `E02_CartPoleUnity`, `E04_BallBalance` | бандиты; Q-learning с дискретизацией; A2C/PPO на непрерывных |
| Б2 | `E05_FoodCollector`, `E06_Hunter3D`, `E07_RacingCar` | REINFORCE; PPO+GAE+PBRS; SAC |
| Б3 | `E08_SoccerArena`, `E09_CurriculumMaze`, `E10_Imitation` | MA-POCA/Self-Play; Curriculum+DR; BC/GAIL |
| Б4 | `E11_Research` (+ добор по решению пользователя) | авторская архитектура |

После каждого батча — отчёт и остановка. Каждый пример закрывается по DoD раздела 14
полностью; частично готовые не помечаются готовыми.

### Фаза 4 — Интеграция и завершение

`docs/RESULTS.md` (IQM + 95% CI по ≥3 сидам), перекрёстные ссылки урок ↔ пример,
`docs/feedback_to_lessons.md`, финальный прогон «с чистого листа» по `README.md`
с фиксацией времени.

---

## 4. Вопросы пользователю (5, каждый с предлагаемым значением по умолчанию)

### Q1. Стек: какой Unity и какой Python/ML-Agents ставим?

Блокеры B-1 и B-2. Без ответа Фаза 1 не может начаться.

**Q1a — Unity.** Проект требует 6000.5.4f1, его на машине нет.
> **Предлагаемое по умолчанию:** установить через Unity Hub ровно **6000.5.4f1**
> (сохраняет `ProjectVersion.txt` и `CLAUDE.md` без изменений).
> Альтернатива: перевести проект на установленный **6000.3.10f1** (тогда обновляем
> `ProjectVersion.txt`, `manifest.json` и документацию; `com.unity.ml-agents` 4.0.3
> требует Unity 6000.0+, так что технически это допустимо).

**Q1b — Python.** Нужен 3.10.1–3.10.12; на машине только 3.11.4.
> **Предлагаемое по умолчанию:** установить **Python 3.10.11**, создать `python/.venv`,
> поставить `mlagents` + `mlagents-envs` **из git по тегу `release_23_tag`**
> (соответствует C#-пакету 4.0.x), `torch` с CUDA 12.1 под GTX 1650.
> Альтернатива: `release_22` (как в уроке 1.2 курса) — но тогда C#-пакет придётся
> откатить с 4.0.3 до 3.0.x.

### Q2. Объём карты примеров: по фактическим урокам или по таксономии инструкции?

Курс содержит 21 урок + 3 проекта; таксономия 13.3 инструкции шире (SARSA, MC,
n-step, Dyna-Q, DDPG, TD3, PER, Rainbow, C51/QR-DQN, A3C в уроках отсутствуют).
> **Предлагаемое по умолчанию:** реализуем **12 примеров `E00`–`E11`** строго по
> фактическим урокам (`docs/02_LESSON_MAP.md`, §2); отсутствующие в курсе методы
> помечаем `NO_LESSON` и не делаем.
> Альтернатива: закрыть всю таксономию 13.3 — это ≈+6 сред и ≈+10 алгоритмов.

### Q3. Куда переносим Unity-проект?

Инструкция (раздел 5) требует `unity/MLAgentsLab/`. Последний коммит репозитория
(`c35e558`) намеренно сделал Unity-проектом **корень**, и это зафиксировано в `CLAUDE.md`.
> **Предлагаемое по умолчанию:** выполнить перенос **в `unity/MLAgentsLab/`** — как
> требует инструкция; это освобождает корень под `python/`, `notebooks/`, `configs/`,
> `scripts/` и снимает риск, что Unity начнёт импортировать Python-файлы как ассеты.
> Альтернатива: оставить Unity-проект в корне (тогда `Assets/Envs/E##_…` кладём в
> корневой `Assets/`, а расхождение с разделом 5 фиксируем ADR-ом).

### Q4. Пилот Фазы 2 — подтверждаете пару `E01_GridWorld` + `E03_RollerBall`?

> **Предлагаемое по умолчанию:** да, как рекомендует инструкция (6/Фаза 2, п. 2):
> `E01` проверяет связку Python↔Unity и корректность MDP, `E03` — полный путь
> ONNX→Unity на DQN. Для `E03` включаем дискретный вариант действий (в коде агента
> он уже заготовлен), непрерывный вариант оставляем для PPO/SAC в Фазе 3.

### Q5. Что делаем со штатным `mlagents-learn` и с существующими конфигами?

Пункт 12.5 — `SHOULD` хранить эталонный конфиг штатного тренера для сравнения.
Существующий `config/RollerAgent.yaml` (PPO) — единственный такой файл.
> **Предлагаемое по умолчанию:** сохраняем эталон: переносим его в
> `configs/mlagents/E03_RollerBall.yaml` (с переименованием behavior name),
> и для каждой среды заводим такой файл. Наши собственные конфиги живут отдельно
> в `configs/E##_<Name>__<algo>.yaml`.
> Альтернатива: отказаться от штатного тренера полностью и убрать конфиг в `_archive/`.

---

## 4a. Ответы пользователя (получены 2026-08-15)

| Вопрос | Решение пользователя |
|---|---|
| **Q1a — Unity** | **Установить 6000.5.4f1.** `ProjectVersion.txt`, `manifest.json` и `CLAUDE.md` остаются без изменений. Фаза 1 начинается после того, как редактор появится в Unity Hub |
| **Q1b — Python / ML-Agents** | **Python 3.10.11 + `python/.venv`**; `mlagents` и `mlagents-envs` из git по тегу **`release_23_tag`**; `torch` с CUDA 12.1 под GTX 1650. `com.unity.ml-agents` остаётся 4.0.3 |
| **Q2 — Объём** | **12 примеров `E00`–`E11`** строго по фактическим урокам. Методы, отсутствующие в курсе (SARSA, MC, n-step, Dyna-Q, DDPG, TD3, PER, Rainbow, C51/QR-DQN, A3C), помечаются `NO_LESSON` и не реализуются |
| **Q3 — Структура** | **Перенос Unity-проекта в `unity/MLAgentsLab/`** через `git mv`, как требует раздел 5 инструкции. Расхождение с коммитом `c35e558` снимается: `CLAUDE.md` и `README.md` обновляются в Фазе 1 |
| **Q5 — штатный тренер** | Ответ не давался → применяется предложенное по умолчанию: эталонные конфиги `mlagents-learn` сохраняются в `configs/mlagents/E##_<Name>.yaml`, собственные — в `configs/E##_<Name>__<algo>.yaml`. Решение отменяемо до начала Фазы 1 |

Q4 (пилот Фазы 2 = `E01_GridWorld` + `E03_RollerBall`) отдельно не переспрашивался —
применяется значение по умолчанию, совпадающее с рекомендацией инструкции 6/Фаза 2.

Зафиксированные следствия для Фазы 1:

- коды примеров `E00`–`E11` считаются **присвоенными** и далее неизменяемы (правило 5.3);
- целевой путь Unity-проекта: `unity/MLAgentsLab/`;
- целевой Python: 3.10.11 в `python/.venv`;
- блокер B-1 снимается пользователем (установка Unity 6000.5.4f1), B-2 — мной в Фазе 1.

---

## 5. Ограничение (снято)

Переход к Фазе 1 требовал ответа на Q1: не было ни Unity нужной версии, ни Python-стека.
Оба блокера сняты — см. раздел 6.

---

## 6. Состояние Фазы 1 (обновляется по ходу)

**Дата обновления:** 2026-08-15.

| Пункт плана Фазы 1 | Статус | Чем подтверждено |
|---|---|---|
| 1. Коммит-снимок перед реструктуризацией | ✔ | коммит `9ac31f2` |
| 2. Реструктуризация по разделу 5 | ✔ | коммит `c38602d`; `_archive/2026-08-15/` |
| 3. Python-окружение, `pip check`, lock-файл | ✔ | `docs/01_STACK.md` §6.1–6.3 |
| 4. Пересверка контракта ONNX с пакетом | ✔ | `docs/04_ONNX_CONTRACT.md` — статус `VERIFIED`, найдены 3 расхождения |
| 5. Пакет `labrl` | ✔ (кроме `algos`/`buffers` — Фаза 2, A-15) | `python/labrl/` |
| 6. Unity `Shared/`: Core, BuildScript, SceneValidator, OnnxContractCheck | ✔ компилируется, `SceneValidator` и `BuildScript` отработали | `docs/01_STACK.md` §6.6 |
| 7. Шаблоны и `00_setup_check.ipynb` | ✔ | `docs/templates/`, `notebooks/` |
| 8. `pytest` зелёный | ✔ | 69 passed, `docs/01_STACK.md` §6.4 |
| 9. Документы каркаса | ✔ | `03_CONVENTIONS`, `05_TENSORBOARD`, `06_WORKFLOW`, `07_TROUBLESHOOTING`, ADR-0001, README, CLAUDE.md |

### Что изменилось относительно ответов на вопросы Фазы 0

| Решение по Q1a | Фактически | Основание |
|---|---|---|
| установить ровно **6000.5.4f1**, `ProjectVersion.txt` не менять | установлен **6000.5.8f1**, `ProjectVersion.txt` обновлён | пользователь установил именно этот патч и дал команду продолжать; `com.unity.ml-agents` 4.0.3 и `com.unity.ai.inference` 2.6.1 не изменились, контракт ONNX не затронут (A-20) |

Прочие отклонения — в `docs/ASSUMPTIONS.md`: A-14 (способ установки ML-Agents),
A-21 (пакеты, поднятые редактором), A-22 (удалён `com.unity.ai.assistant`).

### Гейт Ф1

| Половина гейта | Статус | Подтверждение |
|---|---|---|
| подключение к **headless-билду**, печать размерностей | ✔ **пройдено** | `docs/01_STACK.md` §6.7: `obs_0` формы (25,), ветка действий (4,), 3 завершённых эпизода, различение `terminated`/`truncated` на живой среде |
| подключение к **редактору** | ожидает пользователя | требует нажатия Play в открытом Unity — действие, которое исполнитель выполнить не может |

Чтобы закрыть вторую половину: открыть `unity/MLAgentsLab` в Unity 6000.5.8f1,
открыть сцену `Assets/Envs/E01_GridWorld/Scenes/E01_GridWorld.unity`, запустить
`notebooks/00_setup_check.ipynb` и на ячейке 6 нажать Play в редакторе.

### Что переходит в Фазу 2

1. Привести `E01_GridWorld` и `E03_RollerBall` к стандарту 7.2: `TrainingArea`
   на базе `TrainingAreaBase`, агенты — наследники `AgentBase`, строка-контракт
   в `ENV_SPEC.md`. Сейчас `SceneValidator` даёт по одной ошибке на среду.
2. `labrl/algos` и `labrl/buffers` (A-15), `scripts/train.py`, `scripts/evaluate.py`,
   `scripts/export_onnx.py` (A-16).
3. `ENV_SPEC.md` для `E03_RollerBall` — его нет вовсе.


---

## 7. Состояние Фазы 2 (вертикальный срез)

**Дата обновления:** 2026-08-15.

### Что закрыто

| Пункт | Статус | Подтверждение |
|---|---|---|
| `E01_GridWorld` — табличный Q-learning + Value Iteration | ✔ **DONE** | `docs/envs/E01_GridWorld.md` |
| `E03_RollerBall` — DQN на PyTorch, полный путь ONNX → Unity | ✔ **DONE** | `docs/envs/E03_RollerBall.md` |
| Грабли зафиксированы | ✔ | `docs/07_TROUBLESHOOTING.md`, T-6…T-8 |

### Обе среды приведены к стандарту 7.2

`SceneValidator` по всему проекту: **0 ошибок, 0 предупреждений**. Сделано:
арены на базе `TrainingAreaBase` со своим генератором, агенты — наследники
`AgentBase`, K арен из префаба (4 и 8), корневые группы сцены, строки-контракты
в `ENV_SPEC.md`.

### Что появилось в ядре

- `labrl/algos/tabular/`: `q_learning`, `value_iteration`
- `labrl/algos/dqn.py`: DQN с целевой сетью и Double DQN
- `labrl/buffers/replay.py`: кольцевой буфер воспроизведения
- `labrl/envs/gridworld_mdp.py`: модель MDP для DP и сверки со средой
- `labrl/nets/tabular.py`: таблица Q как экспортируемый линейный слой
- `labrl/train/`: циклы обучения `tabular` и `dqn`
- `scripts/`: `train.py`, `check_inference.py`, `results.py`

### Инструмент проверки требования 10.6

Проверка инференса в Unity автоматизирована и переиспользуема для любой среды:
`InferenceProbe` (рантайм) + `InferenceBuild` (редактор) + `scripts/check_inference.py`.
Скрипт собирает билд с назначенной моделью и `Inference Only`, прогоняет
20 эпизодов и сравнивает награду с Python-оценкой.

### Гейт Ф2

| Часть гейта | Статус |
|---|---|
| агент под управлением ONNX-модели работает в Unity | ✔ подтверждено измерением: `E01` — 20/20 `Goal`, отношение 1.000; `E03` — 20/20 `Goal`, отношение 1.004 |
| **пользователь лично видит это в Unity и графики в TensorBoard** | ожидает пользователя |

Чтобы закрыть вторую часть:

```powershell
# графики
.\scripts	b.ps1

# агент под управлением модели — открыть сцену в редакторе,
# назначить модель из Assets/Envs/<env>/Models/, Behavior Type = Inference Only
```
