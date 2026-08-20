using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E10_Imitation` (TS-010): фиксированный змеевидный коридор
/// 9 × 9 без развилок.
///
/// **Планировка одна и та же всегда** — в этом отличие от
/// `E09_CurriculumMaze`, где каждый эпизод даёт новую карту. Здесь сравниваются
/// три метода (RL, BC, GAIL) на одном бюджете, и раскладка обязана быть
/// одинаковой: иначе сравнение сравнивало бы удачу генерации.
///
/// Раскладка задаётся правилом, а не таблицей: ряд `y` нечётный — сплошная
/// стена с одним проходом, справа при `(y/2) % 2 == 0` и слева иначе.
/// Кратчайший путь от (0, 0) до (8, 8) — 48 ходов.
/// </summary>
public class CorridorArea : TrainingAreaBase
{
    /// <summary>Размер сетки. Фиксирован: рандомизации в этой среде нет.</summary>
    public const int GridSize = 9;

    /// <summary>Размер клетки в единицах Unity.</summary>
    public const float CellSize = 2f;

    /// <summary>Длина кратчайшего пути в ходах. Проверяется тестом сцены.</summary>
    public const int ShortestPath = 48;

    [Header("Ссылки")]
    public Transform agent;
    public Transform goal;

    [Header("Геометрия")]
    [Tooltip("Высота, на которой стоят агент и выход.")]
    public float spawnHeight = 0.5f;

    /// <summary>Клетка агента (столбец, строка).</summary>
    public Vector2Int AgentCell { get; private set; }

    /// <summary>Клетка выхода. Всегда правый верхний угол.</summary>
    public static Vector2Int GoalCell => new Vector2Int(GridSize - 1, GridSize - 1);

    /// <summary>Стартовая клетка. Всегда левый нижний угол.</summary>
    public static Vector2Int StartCell => new Vector2Int(0, 0);

    static readonly Vector2Int[] Directions =
    {
        new Vector2Int(0, 1),   // 0 — север
        new Vector2Int(0, -1),  // 1 — юг
        new Vector2Int(1, 0),   // 2 — восток
        new Vector2Int(-1, 0),  // 3 — запад
    };

    /// <summary>
    /// Стена ли клетка. Правило раскладки живёт здесь и только здесь:
    /// и Setup-скрипт, и агент спрашивают его, а не хранят свою копию.
    /// </summary>
    public static bool IsWallCell(int x, int y)
    {
        if (x < 0 || y < 0 || x >= GridSize || y >= GridSize) return true;
        if (y % 2 == 0) return false;                     // открытый ряд

        // Нечётный ряд — стена с одним проходом: справа, слева, справа, слева…
        int gap = (y / 2) % 2 == 0 ? GridSize - 1 : 0;
        return x != gap;
    }

    /// <summary>Стена ли клетка (метод экземпляра — для удобства агента).</summary>
    public bool IsWall(int x, int y) => IsWallCell(x, y);

    /// <summary>Мировая (локальная относительно арены) позиция центра клетки.</summary>
    public static Vector3 CellToLocal(Vector2Int cell, float height)
    {
        float offset = (GridSize - 1) * 0.5f;
        return new Vector3((cell.x - offset) * CellSize, height, (cell.y - offset) * CellSize);
    }

    /// <summary>Ставит агента на старт. Выход неподвижен.</summary>
    public void ResetEpisode()
    {
        AgentCell = StartCell;
        agent.localPosition = CellToLocal(AgentCell, spawnHeight);
        if (goal != null) goal.localPosition = CellToLocal(GoalCell, spawnHeight);
    }

    /// <summary>Перемещает агента, если целевая клетка проходима.</summary>
    /// <returns>true, если ход состоялся; false — упёрся в стену.</returns>
    public bool TryMove(int direction)
    {
        Vector2Int next = AgentCell + Directions[direction];
        if (IsWall(next.x, next.y)) return false;

        AgentCell = next;
        agent.localPosition = CellToLocal(next, spawnHeight);
        return true;
    }

    /// <summary>Достиг ли агент выхода.</summary>
    public bool AtGoal => AgentCell == GoalCell;

    /// <summary>
    /// Доля пройденного пути: сколько ходов кратчайшего маршрута осталось
    /// позади. Метрика прогресса, а не награда — доля успехов у RL долго
    /// равна нулю, и без неё непонятно, идёт ли обучение вообще.
    ///
    /// Считается по номеру ряда и положению в нём, а не по манхэттену:
    /// в змеевидном коридоре манхэттеново расстояние до выхода не монотонно
    /// вдоль маршрута и на обратном пробеге растёт.
    /// </summary>
    public float Progress
    {
        get
        {
            int row = AgentCell.y;
            // Полностью пройденные ряды: каждый открытый ряд — 8 ходов вдоль
            // плюс 2 хода на переход к следующему.
            int completed = (row / 2) * (GridSize - 1 + 2);
            // Продвижение внутри текущего ряда: чётные ряды проходятся вправо,
            // нечётные (переходные) уже посчитаны переходом.
            int within = row % 2 == 0
                ? ((row / 2) % 2 == 0 ? AgentCell.x : GridSize - 1 - AgentCell.x)
                : GridSize - 1;
            return Mathf.Clamp01((completed + within) / (float)ShortestPath);
        }
    }
}
