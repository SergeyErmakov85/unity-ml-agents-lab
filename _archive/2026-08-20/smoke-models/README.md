# Модели смоук-прогонов, убранные из проекта Unity

**Дата:** 2026-08-20

Четыре `.onnx`, попавшие в `unity/MLAgentsLab/Assets/Envs/*/Models/`
при проверке конвейеров команд:

```
scripts/train.py --config configs/E10_Imitation__bc.yaml           --seed 0 --quick
scripts/train.py --config configs/E10_Imitation__gail.yaml         --seed 0 --quick
scripts/train.py --config configs/E11_Research__fca_ppo.yaml       --seed 0 --quick
scripts/train.py --config configs/E11_Research__ppo_discrete.yaml  --seed 0 --quick
```

## Почему убраны

Это результат **смоук-тестов**, а не обучения: бюджет сокращён вдесятеро
(`--quick`). Оставить их в проекте значило бы выдать смоук-версию
за обученную модель — при том, что документация прямо говорит, что обучение
на `E08`–`E11` не запускалось.

Для `E10_Imitation_bc.onnx` это менее очевидно: BC сходится за 2000
мини-батчей и на смоук-бюджете уже даёт 100 % пройденных коридоров.
Но и он убран — по тому же основанию: происхождение файла важнее его
качества.

## Причина, по которой они вообще там оказались

`scripts/train.py` копировал модель в проект Unity **безусловно**, тогда
как ноутбуки эту защиту имели с самого начала (`if not QUICK_RUN`).
Это ровно та ловушка, которая описана как T-14 в
`docs/07_TROUBLESHOOTING.md`, — и скрипт ей был подвержен.

Исправлено в том же коммите: `_export` и `_export_selfplay` получили
параметр `quick` и на быстром прогоне оставляют модель только в каталоге
прогона (`results/…/onnx/policy.onnx`).

## Как получить настоящие модели

```powershell
$py = ".\python\.venv\Scripts\python.exe"
& $py scripts\train.py --config configs\E10_Imitation__bc.yaml --all-seeds
& $py scripts\train.py --config configs\E10_Imitation__gail.yaml --all-seeds
& $py scripts\train.py --config configs\E11_Research__fca_ppo.yaml --all-seeds
& $py scripts\train.py --config configs\E11_Research__ppo_discrete.yaml --all-seeds
```

Полный прогон скопирует модель в проект сам.
