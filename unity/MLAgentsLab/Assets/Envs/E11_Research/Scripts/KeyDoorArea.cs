using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E11_Research` (TS-011): комната 9 × 9, разделённая стеной
/// с одной дверью; ключ слева, цель справа.
///
/// Задача **иерархична**: сначала «взять ключ», потом «дойти до двери»,
/// потом «дойти до цели». Без ключа правая половина недостижима физически,
/// поэтому «случайно дойти» нельзя, и разница между методами измеряется
/// честно.
///
/// Дверь — тот же куб, что и стена, только выключаемый. Отдельного объекта
/// «открытая дверь» нет: состояние двери — булев флаг арены, а не сцена.
/// </summary>
public class KeyDoorArea : TrainingAreaBase
{
    /// <summary>Размер сетки. Нечётный: столбец стены ровно посередине.</summary>
    public const int GridSize = 9;

    /// <summary>Столбец, по которому идёт разделительная стена.</summary>
    public const int WallColumn = GridSize / 2;   // 4

    /// <summary>Строка, в которой находится дверь.</summary>
    public const int DoorRow = GridSize / 2;      // 4

    /// <summary>Размер клетки в единицах Unity.</summary>
    public const float CellSize = 2f;

    [Header("Ссылки")]
    public Transform agent;
    public Transform key;
    public Transform goal;

    [Tooltip("Куб двери. Выключается, когда ключ подобран.")]
    public GameObject door;

    [Header("Геометрия")]
    [Tooltip("Высота, на которой стоят агент, ключ и цель.")]
    public float spawnHeight = 0.5f;

    /// <summary>Клетка агента (столбец, строка).</summary>
    public Vector2Int AgentCell { get; private set; }

    /// <summary>Клетка ключа.</summary>
    public Vector2Int KeyCell { get; private set; }

    /// <summary>Клетка цели.</summary>
    public Vector2Int GoalCell { get; private set; }

    /// <summary>Клетка двери. Неподвижна.</summary>
    public static Vector2Int DoorCell => new Vector2Int(WallColumn, DoorRow);

    /// <summary>Подобран ли ключ. Он же признак «дверь открыта».</summary>
    public bool HasKey { get; private set; }

    static readonly Vector2Int[] Directions =
    {
        new Vector2Int(0, 1),   // 0 — север
        new Vector2Int(0, -1),  // 1 — юг
        new Vector2Int(1, 0),   // 2 — восток
        new Vector2Int(-1, 0),  // 3 — запад
    };

    /// <summary>
    /// Стена ли клетка. Дверь считается стеной, **пока нет ключа** —
    /// в этом вся задача.
    /// </summary>
    public bool IsWall(int x, int y)
    {
        if (x < 0 || y < 0 || x >= GridSize || y >= GridSize) return true;
        if (x != WallColumn) return false;
        // Столбец стены: проходима только клетка двери и только с ключом.
        return !(y == DoorRow && HasKey);
    }

    /// <summary>Мировая (локальная относительно арены) позиция центра клетки.</summary>
    public static Vector3 CellToLocal(Vector2Int cell, float height)
    {
        float offset = (GridSize - 1) * 0.5f;
        return new Vector3((cell.x - offset) * CellSize, height, (cell.y - offset) * CellSize);
    }

    /// <summary>
    /// Текущая подцель: ключ, если его нет; дверь, если ключ есть, но агент
    /// слева; цель, если агент справа.
    ///
    /// Это подсказка **этапа**, а не маршрута: дойти до подцели всё равно
    /// нужно, обходя стену. Без неё задача превращается в задачу на разведку,
    /// а исследуется здесь структура (ТЗ §15, TS11-4).
    /// </summary>
    public Vector2Int SubGoalCell
    {
        get
        {
            if (!HasKey) return KeyCell;
            return AgentCell.x < WallColumn ? DoorCell : GoalCell;
        }
    }

    /// <summary>Готовит новый эпизод: расставляет агента, ключ и цель.</summary>
    public void ResetEpisode()
    {
        HasKey = false;
        if (door != null) door.SetActive(true);
        if (key != null) key.gameObject.SetActive(true);

        // Агент и ключ — в левой половине, цель — в правой. Половины
        // не пересекаются, поэтому эпизод всегда требует обоих этапов.
        AgentCell = RandomCellInLeftHalf();
        do
        {
            KeyCell = RandomCellInLeftHalf();
        }
        while (KeyCell == AgentCell);

        GoalCell = new Vector2Int(
            NextInt(WallColumn + 1, GridSize),
            NextInt(0, GridSize));

        agent.localPosition = CellToLocal(AgentCell, spawnHeight);
        key.localPosition = CellToLocal(KeyCell, spawnHeight);
        goal.localPosition = CellToLocal(GoalCell, spawnHeight);
    }

    Vector2Int RandomCellInLeftHalf() =>
        new Vector2Int(NextInt(0, WallColumn), NextInt(0, GridSize));

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

    /// <summary>Подбирает ключ, если агент встал на его клетку.</summary>
    /// <returns>true, если ключ подобран именно сейчас.</returns>
    public bool TryPickUpKey()
    {
        if (HasKey || AgentCell != KeyCell) return false;

        HasKey = true;
        if (key != null) key.gameObject.SetActive(false);
        // Дверь исчезает вместе с подбором ключа: состояние двери —
        // это и есть наличие ключа, отдельного объекта «открытая дверь» нет.
        if (door != null) door.SetActive(false);
        return true;
    }

    /// <summary>Достиг ли агент цели. Без ключа цель недостижима физически.</summary>
    public bool AtGoal => AgentCell == GoalCell;

    /// <summary>Манхэттеново расстояние от агента до клетки.</summary>
    public int DistanceTo(Vector2Int cell) =>
        Mathf.Abs(cell.x - AgentCell.x) + Mathf.Abs(cell.y - AgentCell.y);
}
