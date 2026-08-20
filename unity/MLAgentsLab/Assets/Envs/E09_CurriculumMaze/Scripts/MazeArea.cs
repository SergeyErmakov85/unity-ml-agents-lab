using System.Collections.Generic;
using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E09_CurriculumMaze` (TS-009): генерация лабиринта по сложности,
/// пришедшей из Python, и расстановка агента и выхода.
///
/// Три вещи, которые здесь важнее всего:
///
/// 1. **Сложность приходит извне.** `difficulty ∈ [0, 1]` читается базой
///    (`TrainingAreaBase`) из `EnvironmentParametersChannel`; из неё выводится
///    и размер сетки, и плотность стен, и минимальная длина пути. Учебный план
///    живёт в Python и просто меняет это число.
///
/// 2. **Каждый эпизод — новый лабиринт.** Это и есть Domain Randomization:
///    при неизменной планировке агент выучивает **её**, а не умение проходить
///    лабиринты, и на новой карте рассыпается.
///
/// 3. **Проходимость проверяется, а не предполагается.** Случайная расстановка
///    стен регулярно отрезает выход. Среда, изредка выдающая нерешаемый эпизод,
///    портит и обучение, и оценку — причём незаметно: доля успехов просто
///    упирается в потолок, и понять почему нельзя.
///
/// Производительность: стены — пул на максимальный размер сетки; объекты
/// включаются и выключаются, а не создаются и уничтожаются (требование 7.5).
/// </summary>
public class MazeArea : TrainingAreaBase
{
    /// <summary>Максимальный размер сетки. Он же размер пула стен.</summary>
    public const int MaxGridSize = 11;

    /// <summary>Минимальный размер сетки — при нулевой сложности.</summary>
    public const int MinGridSize = 5;

    /// <summary>Размер клетки в единицах Unity.</summary>
    public const float CellSize = 2f;

    [Header("Ссылки")]
    public Transform agent;
    public Transform goal;

    [Tooltip("Родитель пула стен. Заполняется Setup-скриптом.")]
    public Transform wallRoot;

    [Tooltip("Родитель рамки. Масштабируется под текущий размер сетки.")]
    public Transform borderRoot;

    [Header("Сложность")]
    [Tooltip("Доля клеток-стен при difficulty = 0.")]
    [Range(0f, 0.5f)]
    public float wallDensityMin = 0.10f;

    [Tooltip("Доля клеток-стен при difficulty = 1.")]
    [Range(0f, 0.6f)]
    public float wallDensityMax = 0.35f;

    [Tooltip("Сколько раз пробовать сгенерировать проходимый лабиринт, " +
             "прежде чем начать прореживать стены.")]
    public int maxGenerationAttempts = 32;

    [Header("Геометрия")]
    [Tooltip("Высота, на которой стоят агент и выход.")]
    public float spawnHeight = 0.5f;

    /// <summary>Текущий размер сетки. Меняется на границе эпизода.</summary>
    public int GridSize { get; private set; } = MinGridSize;

    /// <summary>Клетка агента (столбец, строка).</summary>
    public Vector2Int AgentCell { get; private set; }

    /// <summary>Клетка выхода.</summary>
    public Vector2Int GoalCell { get; private set; }

    /// <summary>Длина кратчайшего пути от старта до выхода, в ходах.</summary>
    public int ShortestPath { get; private set; }

    // walls[x, y] == true -> клетка непроходима.
    readonly bool[,] m_Walls = new bool[MaxGridSize, MaxGridSize];
    readonly int[,] m_Distance = new int[MaxGridSize, MaxGridSize];
    // Очередь поиска в ширину, предвыделенная: BFS выполняется каждый эпизод
    // в каждой арене, и создавать очередь заново — лишние аллокации.
    readonly Queue<Vector2Int> m_Queue = new Queue<Vector2Int>(MaxGridSize * MaxGridSize);
    readonly List<Vector2Int> m_Free = new List<Vector2Int>(MaxGridSize * MaxGridSize);

    GameObject[,] m_WallPool;

    static readonly Vector2Int[] Directions =
    {
        new Vector2Int(0, 1),   // 0 — север
        new Vector2Int(0, -1),  // 1 — юг
        new Vector2Int(1, 0),   // 2 — восток
        new Vector2Int(-1, 0),  // 3 — запад
    };

    /// <summary>Размер сетки, соответствующий сложности. Всегда нечётный.</summary>
    public int GridSizeFor(float difficulty)
    {
        int steps = Mathf.RoundToInt(Mathf.Clamp01(difficulty) * (MaxGridSize - MinGridSize) / 2f);
        return MinGridSize + 2 * steps;
    }

    /// <summary>Доля клеток-стен, соответствующая сложности.</summary>
    public float WallDensityFor(float difficulty) =>
        Mathf.Lerp(wallDensityMin, wallDensityMax, Mathf.Clamp01(difficulty));

    /// <summary>Проходима ли клетка. Клетка вне сетки считается стеной.</summary>
    public bool IsWall(int x, int y)
    {
        if (x < 0 || y < 0 || x >= GridSize || y >= GridSize) return true;
        return m_Walls[x, y];
    }

    /// <summary>Мировая (локальная относительно арены) позиция центра клетки.</summary>
    public Vector3 CellToLocal(Vector2Int cell)
    {
        float offset = (GridSize - 1) * 0.5f;
        return new Vector3((cell.x - offset) * CellSize, spawnHeight, (cell.y - offset) * CellSize);
    }

    protected override void OnAreaInitialized()
    {
        EnsurePool();
    }

    void EnsurePool()
    {
        if (m_WallPool != null || wallRoot == null) return;

        m_WallPool = new GameObject[MaxGridSize, MaxGridSize];
        foreach (Transform child in wallRoot)
        {
            // Имя задано Setup-скриптом как "Cell_x_y" — иначе восстановить
            // координату клетки по объекту нечем.
            var parts = child.name.Split('_');
            if (parts.Length != 3) continue;
            if (int.TryParse(parts[1], out int x) && int.TryParse(parts[2], out int y))
                m_WallPool[x, y] = child.gameObject;
        }
    }

    /// <summary>
    /// Готовит новый эпизод: читает сложность, генерирует проходимый лабиринт,
    /// ставит агента и выход.
    /// </summary>
    public void ResetEpisode()
    {
        EnsurePool();

        // Сложность применяется НА ГРАНИЦЕ эпизода: смена размера сетки
        // под ногами агента разорвала бы MDP.
        InitializeArea();
        GridSize = GridSizeFor(Difficulty);

        GenerateSolvableMaze();
        ApplyWallsToPool();
        ApplyBorder();

        agent.localPosition = CellToLocal(AgentCell);
        goal.localPosition = CellToLocal(GoalCell);

        MetricsRecorder.MostRecent("Difficulty", Difficulty);
        MetricsRecorder.MostRecent("GridSize", GridSize);
        MetricsRecorder.Histogram("PathLength", ShortestPath);
    }

    /// <summary>Перемещает агента, если целевая клетка проходима.</summary>
    /// <returns>true, если ход состоялся; false — упёрся в стену.</returns>
    public bool TryMove(int direction)
    {
        Vector2Int next = AgentCell + Directions[direction];
        if (IsWall(next.x, next.y)) return false;

        AgentCell = next;
        agent.localPosition = CellToLocal(next);
        return true;
    }

    /// <summary>Достиг ли агент выхода.</summary>
    public bool AtGoal => AgentCell == GoalCell;

    // --- генерация ------------------------------------------------------

    /// <summary>
    /// Генерирует лабиринт, в котором выход достижим и не ближе
    /// <c>GridSize − 1</c> ходов от старта.
    ///
    /// Схема простая и намеренно не «правильный» алгоритм лабиринта
    /// (Prim, Kruskal, recursive backtracker): те дают идеальные лабиринты
    /// без циклов, а нам нужна перестраиваемая по одному числу сложность
    /// и открытые пространства на лёгких уровнях. Случайная плотность плюс
    /// проверка проходимости даёт и то и другое.
    /// </summary>
    void GenerateSolvableMaze()
    {
        float density = WallDensityFor(Difficulty);
        int minSeparation = GridSize - 1;

        for (int attempt = 0; attempt < maxGenerationAttempts; attempt++)
        {
            // На поздних попытках стены прореживаются: при высокой плотности
            // и маленькой сетке требование по дистанции может быть невыполнимо,
            // и лучше выдать чуть более лёгкий лабиринт, чем зациклиться.
            float relaxed = density * (1f - 0.5f * attempt / maxGenerationAttempts);
            FillWalls(relaxed);

            if (!PickCells(minSeparation, out var start, out var goalCell)) continue;

            AgentCell = start;
            GoalCell = goalCell;
            ShortestPath = m_Distance[goalCell.x, goalCell.y];
            return;
        }

        // Аварийный вариант: пустая сетка. Гарантированно проходима, и лучше
        // выдать тривиальный эпизод, чем повесить редактор в бесконечном цикле.
        FillWalls(0f);
        AgentCell = new Vector2Int(0, 0);
        GoalCell = new Vector2Int(GridSize - 1, GridSize - 1);
        BreadthFirst(AgentCell);
        ShortestPath = m_Distance[GoalCell.x, GoalCell.y];
    }

    void FillWalls(float density)
    {
        for (int x = 0; x < GridSize; x++)
            for (int y = 0; y < GridSize; y++)
                m_Walls[x, y] = NextFloat() < density;
    }

    /// <summary>
    /// Выбирает старт и выход: обе клетки проходимы, выход достижим и лежит
    /// не ближе <paramref name="minSeparation"/> ходов.
    /// </summary>
    bool PickCells(int minSeparation, out Vector2Int start, out Vector2Int goalCell)
    {
        start = goalCell = Vector2Int.zero;

        m_Free.Clear();
        for (int x = 0; x < GridSize; x++)
            for (int y = 0; y < GridSize; y++)
                if (!m_Walls[x, y]) m_Free.Add(new Vector2Int(x, y));

        if (m_Free.Count < 2) return false;

        start = m_Free[NextInt(0, m_Free.Count)];
        BreadthFirst(start);

        // Из достижимых клеток берётся самая дальняя, если ближе минимума
        // ничего не нашлось: так лабиринт остаётся содержательным даже когда
        // случайная расстановка отрезала половину сетки.
        Vector2Int best = start;
        int bestDistance = -1;
        foreach (var cell in m_Free)
        {
            int distance = m_Distance[cell.x, cell.y];
            if (distance > bestDistance)
            {
                bestDistance = distance;
                best = cell;
            }
        }

        if (bestDistance < minSeparation) return false;

        goalCell = best;
        return true;
    }

    /// <summary>
    /// Поиск в ширину от клетки: заполняет <see cref="m_Distance"/> длиной
    /// кратчайшего пути, −1 для недостижимых.
    /// </summary>
    void BreadthFirst(Vector2Int from)
    {
        for (int x = 0; x < GridSize; x++)
            for (int y = 0; y < GridSize; y++)
                m_Distance[x, y] = -1;

        m_Queue.Clear();
        m_Distance[from.x, from.y] = 0;
        m_Queue.Enqueue(from);

        while (m_Queue.Count > 0)
        {
            var cell = m_Queue.Dequeue();
            int next = m_Distance[cell.x, cell.y] + 1;

            for (int d = 0; d < Directions.Length; d++)
            {
                var neighbour = cell + Directions[d];
                if (IsWall(neighbour.x, neighbour.y)) continue;
                if (m_Distance[neighbour.x, neighbour.y] >= 0) continue;

                m_Distance[neighbour.x, neighbour.y] = next;
                m_Queue.Enqueue(neighbour);
            }
        }
    }

    // --- отображение ----------------------------------------------------

    void ApplyWallsToPool()
    {
        for (int x = 0; x < MaxGridSize; x++)
        {
            for (int y = 0; y < MaxGridSize; y++)
            {
                var go = m_WallPool[x, y];
                if (go == null) continue;

                bool visible = x < GridSize && y < GridSize && m_Walls[x, y];
                if (go.activeSelf != visible) go.SetActive(visible);
                if (visible) go.transform.localPosition = CellToLocal(new Vector2Int(x, y));
            }
        }
    }

    /// <summary>Подгоняет рамку под текущий размер сетки.</summary>
    void ApplyBorder()
    {
        if (borderRoot == null) return;

        float half = GridSize * CellSize * 0.5f;
        float span = GridSize * CellSize + CellSize;

        SetBorder("North", new Vector3(0f, 1f, half + 0.5f * CellSize), new Vector3(span, 2f, CellSize));
        SetBorder("South", new Vector3(0f, 1f, -half - 0.5f * CellSize), new Vector3(span, 2f, CellSize));
        SetBorder("East", new Vector3(half + 0.5f * CellSize, 1f, 0f), new Vector3(CellSize, 2f, span));
        SetBorder("West", new Vector3(-half - 0.5f * CellSize, 1f, 0f), new Vector3(CellSize, 2f, span));
    }

    void SetBorder(string name, Vector3 position, Vector3 scale)
    {
        var side = borderRoot.Find(name);
        if (side == null) return;
        side.localPosition = position;
        side.localScale = scale;
    }
}
