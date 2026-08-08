# Assets/ML-Agents — общие ассеты из официального репозитория

Здесь лежат файлы, скопированные из
[Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents),
ветка **`release_23`**. Пути и `.meta`-файлы (а значит и GUID-ы) сохранены
такими же, как в оригинале, поэтому любую официальную примерную среду можно
скопировать сюда из `Project/Assets/ML-Agents/Examples/<Имя>/` — ссылки на
общие материалы, меши и префабы разрешатся сами.

Лицензия оригинала — Apache 2.0, см. `LICENSE.md` рядом.

## Что скопировано

```
Examples/SharedAssets/
  Scripts/     общие компоненты, на которых построены примеры ML-Agents
  Prefabs/     агент-кубы, платформы, индикаторы направления, мишени
  Meshes/      AgentCube, платформы, символы
  Materials/   материалы и текстуры к ним
Examples/WallJump/Materials/   два материала, на которые ссылаются общие префабы
```

### Ключевые скрипты

| Файл | Зачем нужен при обучении |
|---|---|
| `ProjectSettingsOverrides.cs` | Вешается на объект в сцене. Задаёт `Time.timeScale`, гравитацию, параметры физики и **пробрасывает их через `EnvironmentParameters`** — то, чем `mlagents-learn --time-scale` и curriculum управляют средой. Ставьте в каждую новую сцену. |
| `ModelOverrider.cs` | Позволяет подменить `.onnx`-модель из командной строки (`--mlagents-override-model`) — прогон обученной модели без правки сцены. |
| `AdjustTrainingTimescale.cs` | Динамически снижает timescale, если физика не успевает. |
| `SensorBase.cs` | Базовый класс для собственных сенсоров (`ISensor`). |
| `GroundContact.cs`, `TargetContact.cs`, `CollisionCallbacks.cs` | Обработка касаний земли/цели с наградой и завершением эпизода. |
| `JointDriveController.cs`, `OrientationCubeController.cs`, `DirectionIndicator.cs` | Основа сред с сочленёнными телами (Crawler, Walker, Worm). |
| `TargetController.cs` | Респавн цели с событиями достижения. |
| `Area.cs` | Базовый класс тренировочной зоны (сброс при `ResetArea`). |
| `CameraFollow.cs`, `FlyCamera.cs`, `Monitor.cs` | Отладочные камеры и экранный монитор значений. |

Намеренно **не скопирован** `ModelCarousel.cs` — он требует пакет
`com.unity.recorder`, которого нет в манифесте проекта, и без него не
компилируется.

## Важно: конвейер рендеринга

Проект использует **URP**, а материалы и шейдеры примеров написаны под
Built-in Render Pipeline. Скопированные материалы будут отображаться
пурпурными, пока их не сконвертировать:

**Window → Rendering → Render Pipeline Converter → Built-in to URP →
Material Upgrade**

На код (`Scripts/`) это не влияет — он от конвейера не зависит и работает
как есть.

## Папка `Timers/`

Создаётся самим ML-Agents во время обучения (`<Behavior>_timers.json`).
В git не попадает — добавлена в `.gitignore`.
