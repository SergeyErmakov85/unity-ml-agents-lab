# scripts/

PowerShell-обвязка над `mlagents-learn` и batch-режимом Unity.
Все скрипты запускаются из корня репозитория и сами находят `.venv` и Unity.

| Скрипт | Назначение |
|---|---|
| `setup-python.ps1` | Создаёт `.venv` (Python 3.10.12 через uv) и ставит `mlagents`. Флаги: `-Cuda cu121`, `-Locked`, `-Recreate`. |
| `train.ps1` | Запускает обучение среды. `-List` — показать все среды с конфигами и behavior-именами. |
| `tensorboard.ps1` | TensorBoard по `results/`. |
| `build-scenes.ps1` | Пересобирает сцены сред в batch-режиме Unity + `ProjectBootstrap.ConfigureAndValidate`. `-List`, `-ValidateOnly`. |
| `new-environment.ps1` | Разворачивает новую среду из `tools/env-template`. |
| `_common.ps1` | Общие функции (поиск сред, конфигов, Unity, venv). Подключается остальными скриптами. |

## Быстрый старт

```powershell
scripts\setup-python.ps1                 # один раз
scripts\train.ps1 -List                  # что вообще есть
scripts\train.ps1 Greed_world            # обучение → затем Play в Unity
scripts\tensorboard.ps1                  # метрики
```

Подробности и разбор ошибок — `docs/TRAINING.md`.

## Соглашения, на которые опираются скрипты

- Среда — папка `Assets/ML-ENVIRONMENTS/<NN-Категория>/<Имя>/`.
- Trainer-конфиг — `<Среда>/config/*.yaml`, ключ под `behaviors:` совпадает
  с Behavior Name в сцене.
- Сборщик сцены — пункт меню **`Tools/RL/Build ...`** у статического
  editor-класса. `build-scenes.ps1` находит такие методы сам; прочие пункты
  `Tools/RL/*` (темы, скриншоты) в batch-прогон не попадают.
- Путь к Unity берётся из `ProjectSettings/ProjectVersion.txt`; переопределяется
  переменной окружения `UNITY_EXE`.
- `build-scenes.ps1` требует **закрытого** редактора: открытый Unity держит
  `Temp/UnityLockfile`, и batch-режим не запустится. Скрипт проверяет это
  заранее и выходит с кодом 2. При открытом редакторе те же действия доступны
  из меню `Tools → RL → …`.
