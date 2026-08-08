# __ENV__

Категория: `__CATEGORY__`  ·  Behavior Name: `__BEHAVIOR__`

Заготовка создана скриптом `scripts\new-environment.ps1`.

## Состав

| Путь | Что это |
|---|---|
| `Scripts/__ENV__Agent.cs` | Подкласс `Agent`: наблюдения, действия, награды |
| `Editor/__ENV__Setup.cs` | Генератор сцены (**Tools → RL → Build __ENV__ Scene**) |
| `config/__BEHAVIOR__.yaml` | Конфиг `mlagents-learn` |
| `Scenes/__ENV__.unity` | Сцена (появится после сборки) |

## Порядок работы

1. Собрать сцену:
   ```powershell
   scripts\build-scenes.ps1 __ENV__
   ```
   или в редакторе: **Tools → RL → Build __ENV__ Scene**.

2. Проверить готовность к обучению:
   **Tools → RL → Validate Training Setup**
   (сверяет Behavior Name из сцены с ключами в `config/*.yaml`).

3. Запустить обучение:
   ```powershell
   scripts\train.ps1 __ENV__ -RunId __ENV_LOWER__-01
   ```
   Дождаться `Listening on port 5004` и нажать **Play** в Unity.

4. Метрики: `scripts\tensorboard.ps1`.

5. Готовая модель — `results\__ENV_LOWER__-01\__BEHAVIOR__.onnx`.
   Перетащить в **Behavior Parameters → Model**, режим **Inference Only**.

## Что менять под свою задачу

- Наблюдения: `CollectObservations` + `BrainParameters.VectorObservationSize`.
- Действия: `OnActionReceived` + `BrainParameters.ActionSpec`.
- Награды: `AddReward` / `SetReward` / `EndEpisode`.

Размерности в агенте и в `__ENV__Setup.cs` должны совпадать — иначе
`mlagents-learn` завершится с ошибкой несовпадения размерностей.
