# 06_WORKFLOW — Сквозной цикл: Unity → build → notebook → ONNX → Unity

**Дата:** 2026-08-15
**Основание:** `CLAUDE_Unity-ml-agents-lab.md`, разделы 6, 7, 9, 10, 14.

Документ описывает путь одного примера от пустой сцены до агента в Unity,
управляемого обученной моделью. Это же — порядок закрытия DoD (раздел 14).

---

## Обзор

```
  ┌──────────────┐   ENV_SPEC.md      ┌──────────────┐
  │ 1. Сцена     │ ─────────────────▶ │ 2. Валидация │
  │  Setup-скрипт│                    │SceneValidator│
  └──────────────┘                    └──────┬───────┘
                                             │ 0 ошибок
                                             ▼
                                      ┌──────────────┐
                                      │ 3. Headless- │
                                      │    сборка    │──▶ builds/E##_<Name>/
                                      └──────┬───────┘
                                             ▼
  ┌──────────────┐                    ┌──────────────┐
  │ 6. Инференс  │◀── .onnx ──────────│ 4. Обучение  │──▶ results/…/tb/
  │   в Unity    │                    │  (ноутбук)   │
  └──────────────┘                    └──────┬───────┘
         ▲                                   ▼
         │                            ┌──────────────┐
         └────── ≥ 0.8 × Python ──────│ 5. Экспорт + │
                                      │  верификация │
                                      └──────────────┘
```

---

## Шаг 1. Сцена

Сцена **строится кодом**, а не собирается мышью: `Assets/Envs/E##_<Name>/Editor/<Name>Setup.cs`
с пунктом меню `Tools/RL/…`. Чтобы изменить сцену, меняется скрипт и сцена
пересобирается. Так сцена воспроизводима и её изменения читаемы в diff.

```powershell
& "C:\Program Files\Unity\Hub\Editor\6000.5.8f1\Editor\Unity.exe" `
  -batchmode -quit -projectPath C:\unity-ml-agents-lab\unity\MLAgentsLab `
  -executeMethod GridWorldSetup.BuildScene -logFile build.log
```

Обязательное содержимое сцены — требование 7.2: корневые группы, `TrainingArea`
как Prefab, агент с `BehaviorParameters` (Behavior Name = `E##_<Name>`) и
`DecisionRequester`, реализованный `Heuristic()`, явный `MaxStep`.

Спецификация `ENV_SPEC.md` пишется **до** реализации (требование 7.1) и содержит
строку-контракт для валидатора (`docs/03_CONVENTIONS.md`, §3).

## Шаг 2. Валидация

```powershell
& $unity -batchmode -quit -projectPath .\unity\MLAgentsLab `
  -executeMethod LabRL.EditorTools.SceneValidator.ValidateAllBatch -logFile validate.log
```

Ненулевой код возврата = есть ошибки. Проверяется совпадение Behavior Name
с идентификатором среды, соответствие размерностей `ENV_SPEC.md`, наличие
`DecisionRequester`, явный `MaxStep`, отсутствие агентов вне `TrainingArea`
и наличие тегов проекта.

## Шаг 3. Headless-сборка

```powershell
& $unity -batchmode -quit -projectPath .\unity\MLAgentsLab `
  -executeMethod LabRL.EditorTools.BuildScript.BuildEnv `
  -envId E03_RollerBall -outputPath ..\..\builds\E03_RollerBall -logFile build.log
echo $LASTEXITCODE   # 0 = успех
```

Билд нужен для обучения: он запускается с `no_graphics=True` и `time_scale=20`,
что даёт на порядок больше шагов в секунду, чем редактор. Отладка при этом
ведётся в редакторе (`file_name=None`) — там видно, что делает агент.

## Шаг 4. Обучение

Ноутбук `notebooks/E##_<Name>__<algo>.ipynb` со структурой ячеек раздела 9
инструкции. Логика алгоритма **импортируется** из `labrl.algos`, а не копируется
в ноутбук (требование 9.4); прозрачность даёт ячейка 6, печатающая
`inspect.getsource(Algo.update)`.

Режим `QUICK_RUN = True` — сокращённый бюджет, ≤ 5 минут, для проверки и CI.
`QUICK_RUN = False` — полное обучение.

Без ноутбука (CI, прогон по трём сидам). Ключ `--quick` — смоук-тест
конвейера: критерий приёмки при нём не проверяется, потому что на сокращённом
бюджете он и не обязан достигаться.

```powershell
python scripts\train.py --config configs\E03_RollerBall__dqn.yaml --seed 0 --quick

# полный прогон по всем сидам конфига
python scripts\train.py --config configs\E03_RollerBall__dqn.yaml --all-seeds

# два обучения одновременно: разные порты связи с Unity
python scripts\train.py --config configs\E04_BallBalance__ppo.yaml --all-seeds --worker-id 3
```

Результаты — в `results/E##_<Name>/<algo>/<YYYYMMDD-HHMMSS>_seed<k>/`
(структура — `docs/05_TENSORBOARD.md`, §5). Просмотр: `.\scripts\tb.ps1`.

Сводка по всем прогонам с IQM и доверительными интервалами:

```powershell
python scripts\results.py --write docs\RESULTS.md
```

## Шаг 5. Экспорт ONNX и верификация

```python
from labrl.export.onnx_export import export_policy_to_onnx
from labrl.export.onnx_verify import verify_onnx_model

path = export_policy_to_onnx(algo.policy_module(), action_spec, obs_shapes, run.onnx / "policy.onnx")
result = verify_onnx_model(path, algo.policy_module(), action_spec, obs_shapes, sample_obs=obs_batch)
print(result.report())
result.raise_if_failed()      # провал блокирует приёмку (10.5)
```

Верификация проверяет структуру, opset, имена входов и выходов, значения
констант, прогон на батчах 1 и 64, числовой паритет с PyTorch (≤ 1e-4)
и диапазоны действий. Контракт — `docs/04_ONNX_CONTRACT.md`.

`sample_obs` — наблюдения **из реальной среды**, а не случайный шум: расхождение
нормализации или порядка наблюдений проявляется именно на реальном распределении.

## Шаг 6. Возврат в Unity

1. Скопировать `.onnx` в `Assets/Envs/E##_<Name>/Models/`.
2. Дождаться импорта, выделить модель, выполнить `Tools/RL/Check ONNX Contract` —
   это подтверждает, что Inference Engine видит те же имена тензоров.
3. В `BehaviorParameters` агента: `Model` = импортированная модель,
   `Behavior Type` = `Inference Only`.
4. Прогнать **20 эпизодов** и сравнить среднюю награду с Python-оценкой.

Шаги 1–4 автоматизированы и выполняются одной командой:

```powershell
python scripts\check_inference.py --config configs\E03_RollerBall__dqn.yaml --python-reward 0.95
```

Скрипт собирает отдельный билд с назначенной моделью и `Inference Only`,
прогоняет 20 эпизодов и печатает отношение Unity / Python.

Критерий приёмки: награда в Unity ≥ **0.8 ×** награды в Python (требование 10.6).
Ниже порога — расследование, а не «и так сойдёт». Типовые причины и их
разбор — `docs/07_TROUBLESHOOTING.md`.

### Модель в проекте Unity после быстрого прогона

Ноутбук и `train.py` пишут модель по одному и тому же пути, поэтому запуск
ноутбука в режиме `QUICK_RUN = True` затирает модель полного прогона моделью
смоук-теста. Файл при этом остаётся валидным и проходит верификацию, но агент
в Unity ведёт себя хуже, чем указано в карточке среды. Восстановить модели
последних **полных** прогонов:

```powershell
python scripts\sync_models.py            # показать расхождения
python scripts\sync_models.py --apply    # скопировать
```

---

## Что делает пример завершённым

Полный список — раздел 14 инструкции (DoD). Ключевое: ни один пункт не
помечается выполненным без фактического запуска проверки (16.3), полное
обучение проводится минимум на **3 сидах** (12.2), а итоговая метрика примера —
**IQM с доверительным интервалом**, а не результат одного удачного прогона (12.3).
