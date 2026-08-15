# 03_CONVENTIONS — Соглашения проекта

**Дата:** 2026-08-15
**Основание:** `CLAUDE_Unity-ml-agents-lab.md`, разделы 5, 7, 8, 9, 11.

Документ описывает **только то, что обязано совпадать** между Unity, Python,
конфигами и документацией. Всё остальное — вопрос вкуса и здесь не регулируется.

---

## 1. Идентификатор среды `E##_<Name>`

Главное соглашение проекта. Одна строка связывает шесть мест:

| Место | Как выглядит |
|---|---|
| Каталог Unity | `unity/MLAgentsLab/Assets/Envs/E03_RollerBall/` |
| Сцена | `Assets/Envs/E03_RollerBall/Scenes/E03_RollerBall.unity` |
| **Behavior Name** в `BehaviorParameters` | `E03_RollerBall` |
| Каталог сборки | `builds/E03_RollerBall/E03_RollerBall.exe` |
| Каталог результатов | `results/E03_RollerBall/<algo>/…` |
| Конфиг, ноутбук, ENV CARD | `configs/E03_RollerBall__dqn.yaml`, `notebooks/E03_RollerBall__dqn.ipynb`, `docs/envs/E03_RollerBall.md` |

Правила:

- `E` + **две** цифры с ведущим нулём + `_` + имя в `PascalCase` без разделителей;
- код присвоен навсегда: `E03` не переиспользуется, даже если среда удалена;
- **Behavior Name обязан совпадать с идентификатором** (правило 5.3 инструкции).
  Это единственная точка связи Python ↔ Unity: при расхождении Python просто
  не увидит агентов, без внятной ошибки. Проверяется в трёх местах:
  `AgentBase.AssertBehaviorName()` в рантайме, `SceneValidator` перед сборкой,
  `labrl.utils.config` при чтении конфига.

Присвоенные коды: см. `docs/02_LESSON_MAP.md`.

---

## 2. Каталоги среды

```
Assets/Envs/E##_<Name>/
├── ENV_SPEC.md      # ТЗ сцены по Playbook v3.0 + строка-контракт для валидатора
├── Editor/          # <Name>Setup.cs — сцена строится кодом, а не руками
├── Scenes/          # ровно одна сцена: E##_<Name>.unity
├── Scripts/         # <Name>Agent.cs, <Name>Area.cs
├── Prefabs/         # TrainingArea.prefab и прочее
├── Materials/
├── Textures/        # если нужны
└── Models/          # обученные .onnx
```

Ограничения:

- **ровно одна сцена** на среду — `BuildScript` и `SceneValidator` считают
  вторую сцену ошибкой, потому что «собрать какую-нибудь» означает получить
  билд, в котором Python не найдёт нужное поведение;
- все пути ассетов среды остаются внутри её папки (константа `Root`
  в Setup-скрипте);
- сцены **генерируются кодом**: чтобы изменить сцену, меняется Setup-скрипт
  и сцена пересобирается, а не правится руками в редакторе.

---

## 3. Строка-контракт в `ENV_SPEC.md`

`ENV_SPEC.md` — текст для человека. Чтобы `SceneValidator` мог сверить
спецификацию с реализацией (требование 7.7), в неё добавляется одна
машиночитаемая строка:

```markdown
<!-- validator: obs_size=25; discrete_branches=4; continuous_size=0; max_step=100 -->
```

| Ключ | Смысл | Отсутствие ключа |
|---|---|---|
| `obs_size` | `BrainParameters.VectorObservationSize` | не проверяется |
| `discrete_branches` | размеры дискретных веток через запятую; пусто — веток нет | не проверяется |
| `continuous_size` | `ActionSpec.NumContinuousActions` | не проверяется |
| `max_step` | `Agent.MaxStep` | не проверяется |

Отсутствие всей строки — предупреждение валидатора, а не ошибка.

---

## 4. Пространства имён и имена типов C#

Проект **один** на все среды (ADR-0001), поэтому имена типов должны быть
уникальны глобально.

| Что | Правило | Пример |
|---|---|---|
| Общий рантайм-код | `namespace LabRL.Core` | `AgentBase`, `SpawnService` |
| Общий редакторный код | `namespace LabRL.EditorTools` | `BuildScript`, `SceneValidator` |
| Код среды | без пространства имён, но с уникальным префиксом имени | `GridWorldAgent`, `RollerAgent` |
| Setup-скрипт среды | `<Name>Setup` в `Editor/`, пункт меню `Tools/RL/…` | `GridWorldSetup.BuildScene` |

---

## 5. Python: имена и раскладка

| Что | Правило |
|---|---|
| Пакет | `labrl`, живёт в `python/labrl` |
| Модуль алгоритма | один файл на алгоритм: `labrl/algos/dqn.py` |
| Класс алгоритма | `DQN`, конфиг рядом — `DQNConfig` (dataclass) |
| Единственная точка обновления параметров | метод `update(batch) -> dict[str, float]` (требование 8.7) |
| Модуль для экспорта | метод `policy_module() -> nn.Module` |
| Сторонние RL-библиотеки | запрещены в учебных реализациях (16.5) |

Все значения, влияющие на результат, приходят из конфига; «магических чисел»
внутри функций быть не должно (8.6).

---

## 6. Конфиги

| Файл | Что это |
|---|---|
| `configs/E##_<Name>__<algo>.yaml` | **наш** конфиг эксперимента (структура — Приложение A инструкции) |
| `configs/mlagents/E##_<Name>.yaml` | эталонный конфиг штатного `mlagents-learn` для сравнения (п. 12.5) |

В эталонном конфиге ключ `behaviors:` содержит **тот же** идентификатор среды —
иначе штатный тренер не подключится к сцене.

Блок `success_criteria` обязателен: он же критерий приёмки примера (12.4).

Один пример может иметь **несколько** конфигов — по одному на алгоритм
(`E00_Bandit__ucb.yaml`, `E04_BallBalance__ppo.yaml`). Общая часть у них
обязана совпадать: сравнение методов при разных бюджетах или разных сетях
сравнивает не методы, а настройки.

### Параметры-расписания

Ключи `algo.epsilon` и `algo.learning_rate` принимают **либо число** (постоянное
значение), **либо блок расписания**:

```yaml
algo:
  learning_rate:
    type: linear        # constant | linear | exponential
    start: 0.20
    end: 0.01
    decay_steps: 120000
```

Разбор — `labrl.utils.schedules.build_schedule`. Постоянный шаг обучения
у методов, которые сходятся только при убывающем шаге, — сознательный выбор
пользователя, а не значение по умолчанию (T-11, T-13
в `docs/07_TROUBLESHOOTING.md`).

---

## 7. Каталог прогона

```
results/E##_<Name>/<algo>/<YYYYMMDD-HHMMSS>_seed<k>/
├── tb/             логи TensorBoard
├── ckpt/           чекпойнты
├── onnx/           экспортированные модели
├── config.yaml     копия конфига
├── env_info.json   размерности пространств, режим подключения, версия билда
├── metrics.json    сводка метрик
└── pip_freeze.txt  слепок окружения
```

Создаётся `labrl.logging.run_dir.create_run_dir`. Каталог `results/`
в `.gitignore` целиком, кроме `results/README.md`.

---

## 8. Имена метрик TensorBoard

Обязательные теги — в `docs/05_TENSORBOARD.md` и в константах
`labrl.logging.tb_logger.Tags`. Правило одно: **имена не изобретаются на месте**,
берутся из `Tags`. Свои метрики алгоритма идут в `Custom/`, метрики из Unity —
в `Env/`.

---

## 9. Git

- ветка работы: `feature/lab-bootstrap`; прямые коммиты в `main` и force-push запрещены (3.3);
- Conventional Commits с кодом примера: `feat(E03): DQN training loop and ONNX export`,
  `docs(core): ONNX contract`, `chore(repo): restructure directories` (3.4);
- ничего не удаляется безвозвратно — вытесненное уходит в `_archive/<YYYY-MM-DD>/`
  с сохранением относительных путей (3.2);
- `Library/`, `builds/`, `results/`, веса больше 50 МБ не коммитятся (16.6).

---

## 10. Язык

Документация, комментарии и сообщения коммитов — русский. Имена сущностей
(файлы, классы, теги TensorBoard, ключи конфигов) — английский, потому что
теги обязаны совпадать со штатным ML-Agents, а имена файлов — с путями в коде.
