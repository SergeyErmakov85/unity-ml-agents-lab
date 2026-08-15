# 05_TENSORBOARD — Схема логирования

**Дата:** 2026-08-15
**Основание:** `CLAUDE_Unity-ml-agents-lab.md`, раздел 11.
**Реализация:** `python/labrl/logging/tb_logger.py`, `python/labrl/logging/run_dir.py`.

---

## 1. Зачем фиксировать схему

Метрики двенадцати примеров, написанных в разное время, должны ложиться на один
график. Это возможно, только если имена тегов совпадают побуквенно. Поэтому
имена вынесены в константы `labrl.logging.tb_logger.Tags` и **не пишутся строками
по месту**. Второе следствие: наши имена совпадают с тегами штатного
`mlagents-learn`, поэтому «моя реализация» и «эталон» сравниваются в одном
TensorBoard без переименований.

---

## 2. Обязательные скаляры (таблица 11.2)

| Тег | Константа в коде | Смысл | Когда пишется |
|---|---|---|---|
| `Environment/Cumulative Reward` | `Tags.CUMULATIVE_REWARD` | средняя награда за эпизод | по завершении эпизодов |
| `Environment/Episode Length` | `Tags.EPISODE_LENGTH` | средняя длина эпизода | по завершении эпизодов |
| `Losses/Value Loss` | `Tags.VALUE_LOSS` | потери критика / Q | каждое обновление |
| `Losses/Policy Loss` | `Tags.POLICY_LOSS` | потери актора | каждое обновление |
| `Policy/Entropy` | `Tags.ENTROPY` | энтропия политики | каждое обновление |
| `Policy/Learning Rate` | `Tags.LEARNING_RATE` | текущий LR | каждое обновление |
| `Policy/Epsilon` | `Tags.EPSILON` | текущее ε | каждый шаг сбора |
| `Eval/Mean Reward` | `Tags.EVAL_MEAN_REWARD` | оценка детерминированной политики | по расписанию оценки |
| `Eval/Success Rate` | `Tags.EVAL_SUCCESS_RATE` | доля успешных эпизодов | по расписанию оценки |
| `Perf/Steps Per Second` | `Tags.STEPS_PER_SECOND` | пропускная способность | раз в N шагов |

Полнота проверяется методом `TBLogger.missing_required_tags()`: он возвращает
теги, которые в прогоне так и не появились. Пустой результат — обязательный
пункт DoD примера.

Про теги, не имеющие смысла для алгоритма. `Policy/Epsilon` бессмысленно для
PPO, `Losses/Policy Loss` — для табличного Q-learning. Такой тег всё равно
пишется, с постоянным значением (0 или фактическим ε=0), чтобы состав тегов
не зависел от алгоритма и графики сравнивались напрямую.

---

## 3. Метрики алгоритма: неймспейс `Custom/`

Пишутся через `TBLogger.custom(name, value, step)` — префикс добавляется сам.

| Тег | Где применяется |
|---|---|
| `Custom/TD Error` | DQN и производные |
| `Custom/Q Max` | DQN |
| `Custom/Buffer Size` | все методы с replay-буфером |
| `Custom/Grad Norm` | все методы с клиппингом градиента |
| `Custom/Alpha` | SAC |
| `Custom/KL` | PPO |
| `Custom/Clip Fraction` | PPO |

Список открыт: любая метрика, специфичная для алгоритма, идёт сюда.

---

## 4. Метрики из Unity: неймспейс `Env/`

Среда публикует статистики через `Academy.Instance.StatsRecorder.Add(...)`
(обёртка — `LabRL.Core.MetricsRecorder`). Они доходят до Python через
`StatsSideChannel` и логируются с префиксом `Env/`:

| Что публикует среда | Тег в TensorBoard | Метод обёртки |
|---|---|---|
| `SuccessRate` | `Env/SuccessRate` | `MetricsRecorder.Success(bool)` |
| `EpisodeSteps` | `Env/EpisodeSteps` | `MetricsRecorder.Histogram` |
| произвольная метрика среды | `Env/<имя>` | `Average` / `Sum` / `MostRecent` / `Histogram` |

Среда **не знает** о префиксе `Env/`: его добавляет Python-сторона
(`TBLogger.env_stats`). Так среда остаётся независимой от схемы TensorBoard.

Способ агрегации выбирается по смыслу величины: доля успехов — `Average`,
счётчик событий — `Sum`, текущая сложность curriculum — `MostRecent`,
распределение длин эпизодов — `Histogram`.

---

## 5. Что ещё пишется в прогон (требование 11.5)

| Что | Как |
|---|---|
| Гиперпараметры | `TBLogger.hparams(...)` → `add_hparams` |
| Полный конфиг текстом | `TBLogger.text("config", ...)` |
| Сид | в `hparams` и в имени каталога прогона |
| Версии пакетов | `run_dir.write_pip_freeze()` → `pip_freeze.txt` |
| Хэш git-коммита | `tb_logger.git_commit_hash()` → в `hparams` |
| Размерности пространств, режим подключения | `run_dir.write_env_info(...)` → `env_info.json` |

---

## 6. Просмотр

Одна команда на все примеры (требование 11.6):

```powershell
.\scripts\tb.ps1
# эквивалент: tensorboard --logdir results
```

Скрипт использует интерпретатор из `python/.venv`, поэтому TensorBoard
не требуется ставить глобально.
