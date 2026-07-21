using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Дискретная сетка 5×5 для Q-learning (TS-001).
/// Вся логика индексная: клетка (c, r) → state = r·cols + c.
/// Координаты клеток локальны относительно контейнера TrainingArea,
/// поэтому префаб можно инстанцировать со смещением для параллельного обучения.
/// </summary>
public class GridWorldEnvironment : MonoBehaviour
{
    [Header("Grid")]
    public int cols = 5;
    public int rows = 5;
    public float cellSize = 1f;

    [Header("Cells (x = столбец, y = ряд)")]
    public Vector2Int startCell = new Vector2Int(0, 0);
    public Vector2Int goalCell = new Vector2Int(4, 4);
    public List<Vector2Int> trapCells = new List<Vector2Int>
    {
        new Vector2Int(1, 3),
        new Vector2Int(3, 1),
    };
    public List<Vector2Int> wallCells = new List<Vector2Int>
    {
        new Vector2Int(1, 2),
        new Vector2Int(3, 3),
        new Vector2Int(3, 4),
    };

    [Header("Rewards")]
    public float stepReward = -0.04f;
    public float goalReward = 1f;
    public float trapReward = -1f;
    public int maxSteps = 100;

    [Header("Stochasticity")]
    public bool randomStart = false;
    [Range(0f, 1f)] public float slipProbability = 0f;

    // 0 = North (+Z), 1 = South (−Z), 2 = East (+X), 3 = West (−X)
    private static readonly Vector2Int[] Directions =
    {
        new Vector2Int(0, 1),
        new Vector2Int(0, -1),
        new Vector2Int(1, 0),
        new Vector2Int(-1, 0),
    };

    public Vector3 CellToWorld(Vector2Int cell, float y)
    {
        float offsetX = (cols - 1) * 0.5f;
        float offsetZ = (rows - 1) * 0.5f;
        var local = new Vector3((cell.x - offsetX) * cellSize, y, (cell.y - offsetZ) * cellSize);
        return transform.TransformPoint(local);
    }

    public bool InBounds(Vector2Int cell) =>
        cell.x >= 0 && cell.x < cols && cell.y >= 0 && cell.y < rows;

    public bool IsWall(Vector2Int cell) => wallCells.Contains(cell);

    public bool IsTrap(Vector2Int cell) => trapCells.Contains(cell);

    public bool IsGoal(Vector2Int cell) => cell == goalCell;

    public int StateIndex(Vector2Int cell) => cell.y * cols + cell.x;

    /// <summary>
    /// Применяет действие к клетке. Выход за границы или стена — ход блокирован,
    /// возвращается исходная клетка. При slipProbability > 0 действие с этой
    /// вероятностью заменяется одним из двух перпендикулярных (модель FrozenLake).
    /// </summary>
    public Vector2Int ResolveMove(Vector2Int cell, int action)
    {
        if (slipProbability > 0f && Random.value < slipProbability)
        {
            bool vertical = action <= 1;
            action = vertical
                ? (Random.value < 0.5f ? 2 : 3)
                : (Random.value < 0.5f ? 0 : 1);
        }

        Vector2Int next = cell + Directions[action];
        return (!InBounds(next) || IsWall(next)) ? cell : next;
    }

    /// <summary>Равномерный выбор клетки, не являющейся стеной, целью или ловушкой.</summary>
    public Vector2Int RandomFreeCell()
    {
        var free = new List<Vector2Int>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var cell = new Vector2Int(c, r);
                if (!IsWall(cell) && !IsTrap(cell) && !IsGoal(cell))
                    free.Add(cell);
            }
        }
        return free[Random.Range(0, free.Count)];
    }
}
