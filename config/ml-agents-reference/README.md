# Справочные trainer-конфиги ML-Agents

Скопировано из [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)
(ветка `release_23`, папка `config/`) как справочник по гиперпараметрам.

**Это не конфиги сред этого репозитория.** Рабочий конфиг каждой среды лежит
внутри неё: `Assets/ML-ENVIRONMENTS/<Категория>/<Среда>/config/<Behavior>.yaml`.

| Папка | Тренер | Примеры |
|---|---|---|
| `ppo/` | PPO | 3DBall, GridWorld, Walker, Crawler, PushBlock, Hallway, Pyramids… |
| `sac/` | SAC | те же среды с off-policy настройками |
| `poca/` | MA-POCA | кооперативные многоагентные среды (Soccer, DungeonEscape) |
| `imitation/` | GAIL / BC | обучение по демонстрациям |

Полезно подсматривать сюда при настройке новой среды: берите конфиг задачи,
похожей по размеру наблюдений и типу действий.

Описание всех параметров:
<https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Training-Configuration-File.html>
