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

> **Статус:** Фаза 1 (каркас платформы). Готовы: структура репозитория,
> Python-ядро `labrl`, общий Unity-код `Assets/Shared`, контракт ONNX.
> Первые сквозные примеры — Фаза 2. Текущее состояние и открытые вопросы —
> в [`PLAN.md`](PLAN.md).

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
│   └── .venv/                      # изолированное окружение (Python 3.10.11)
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
| Python | **3.10.11** (не 3.11+: `mlagents-envs` требует `<=3.10.12`) | `py -0p` |
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
.\python\.venv\Scripts\python.exe -m pip install --upgrade pip
.\python\.venv\Scripts\python.exe -m pip install -e python
.\python\.venv\Scripts\python.exe -m pytest python\tests -q
```

Зелёный `pytest` означает, что ядро и контракт экспорта ONNX работают.

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
| [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) | реестр допущений |
| [`docs/adr/`](docs/adr/) | архитектурные решения |

## Лицензия

MIT — см. [LICENSE](LICENSE).
