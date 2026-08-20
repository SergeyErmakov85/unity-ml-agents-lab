using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент среды `E11_Research` (TS-011): взять ключ, пройти в дверь, дойти
/// до цели.
///
/// Особенность наблюдения: **первые семь признаков бинарны**, и именно они
/// образуют формальный контекст для FCA (`labrl.utils.fca`). Бинарны они
/// не ради удобства метода, а потому что задача такая: ключ либо есть, либо
/// нет; дверь либо открыта, либо нет; агент либо слева от стены, либо справа.
/// Формальный контекст здесь не выдуман — он взят из описания задачи.
///
/// Порядок признаков 0–6 обязан совпадать с
/// `labrl.nets.fca.KEYDOOR_ATTRIBUTES`: слой понятий индексирует их напрямую,
/// и перестановка сломала бы смысл каждого понятия, не выдав ошибки.
///
/// Модуль курса: `/fca-rl` — «FCA + RL: формальный анализ концептов».
/// </summary>
public class KeyDoorAgent : AgentBase
{
    [Header("Ссылки")]
    public KeyDoorArea area;

    [Header("Награда")]
    [Tooltip("Награда за подобранный ключ. Это отметка пройденного ЭТАПА, " +
             "а не формирование награды: без ключа цель недостижима физически.")]
    public float keyReward = 0.3f;

    [Tooltip("Награда за достижение цели.")]
    public float goalReward = 1f;

    [Tooltip("Штраф за ход в стену. Агент остаётся на месте.")]
    public float wallPenalty = -0.02f;

    [Tooltip("Штраф за шаг. Равен −1/MaxStep.")]
    public float stepPenalty = -1f / 200f;

    public override string EnvId => "E11_Research";

    /// <summary>Не дошёл за отведённое время — это неудача.</summary>
    protected override bool TimeoutIsSuccess => false;

    int m_HeuristicAction = -1;
    bool m_KeyCounted;

    public override void Initialize()
    {
        base.Initialize();
        if (area == null) area = GetComponentInParent<KeyDoorArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetEpisode();
        m_KeyCounted = false;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        const float scale = KeyDoorArea.GridSize - 1;
        var cell = area.AgentCell;
        var subGoal = area.SubGoalCell;

        // --- бинарные признаки: формальный контекст FCA (индексы 0–6) ---
        // Порядок обязан совпадать с labrl.nets.fca.KEYDOOR_ATTRIBUTES.
        sensor.AddObservation(area.HasKey ? 1f : 0f);                                   // 0
        // Дверь открыта тогда и только тогда, когда есть ключ. Признак
        // избыточен по построению — и это сознательно: FCA обязан САМ
        // обнаружить, что два признака эквивалентны, и склеить их
        // в одно понятие. Если не обнаружит — метод не работает.
        sensor.AddObservation(area.HasKey ? 1f : 0f);                                   // 1
        sensor.AddObservation(cell.x > KeyDoorArea.WallColumn ? 1f : 0f);                // 2
        sensor.AddObservation(cell.y > KeyDoorArea.DoorRow ? 1f : 0f);                   // 3
        sensor.AddObservation(cell.y == KeyDoorArea.DoorRow ? 1f : 0f);                  // 4
        int doorDistance = area.DistanceTo(KeyDoorArea.DoorCell);
        sensor.AddObservation(area.DistanceTo(area.KeyCell) < doorDistance ? 1f : 0f);   // 5
        sensor.AddObservation(area.DistanceTo(area.GoalCell) < doorDistance ? 1f : 0f);  // 6

        // --- непрерывные признаки ---
        sensor.AddObservation(cell.x / scale);                                           // 7
        sensor.AddObservation(cell.y / scale);                                           // 8
        sensor.AddObservation((subGoal.x - cell.x) / scale);                             // 9
        sensor.AddObservation((subGoal.y - cell.y) / scale);                             // 10

        // Локальная проходимость. Бинарна, но в контекст НЕ входит: эти
        // признаки зависят от положения и порождали бы понятия вида
        // «я у стены слева», не относящиеся к структуре задачи (ТЗ §4).
        sensor.AddObservation(area.IsWall(cell.x, cell.y + 1) ? 1f : 0f);                // 11
        sensor.AddObservation(area.IsWall(cell.x, cell.y - 1) ? 1f : 0f);                // 12
        sensor.AddObservation(area.IsWall(cell.x + 1, cell.y) ? 1f : 0f);                // 13
        sensor.AddObservation(area.IsWall(cell.x - 1, cell.y) ? 1f : 0f);                // 14

        sensor.AddObservation(MaxStep > 0 ? Mathf.Clamp01((float)StepCount / MaxStep) : 0f);  // 15
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int direction = actions.DiscreteActions[0];

        if (!area.TryMove(direction))
        {
            AddReward(wallPenalty);
            MetricsRecorder.Sum("WallBumps", 1f);
        }

        if (area.TryPickUpKey())
        {
            AddReward(keyReward);
            MetricsRecorder.Sum("KeyPickups", 1f);
            m_KeyCounted = true;
        }

        AddReward(stepPenalty);
        TrackStep();

        if (area.AtGoal)
        {
            AddReward(goalReward);
            MetricsRecorder.Average("KeyRate", 1f);
            EndEpisodeWithResult("Goal", success: true);
        }
        else if (MaxStep > 0 && StepCount >= MaxStep - 1)
        {
            // Доля эпизодов с подобранным ключом — главная диагностика
            // среды: пока она около нуля, обсуждать доли успехов
            // бессмысленно, агент не прошёл первый этап.
            MetricsRecorder.Average("KeyRate", m_KeyCounted ? 1f : 0f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        // −1 означает «клавиша не нажата»: повторять последнее направление
        // нельзя, иначе агент едет сам и проверить среду руками не выйдет.
        discrete[0] = m_HeuristicAction >= 0 ? m_HeuristicAction : 0;
        m_HeuristicAction = -1;
    }

    void Update()
    {
        // Ввод читается в Update, а расходуется в Heuristic: FixedUpdate
        // может пропустить короткое нажатие, а Update — нет.
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.upArrowKey.wasPressedThisFrame || kb.wKey.wasPressedThisFrame) m_HeuristicAction = 0;
        else if (kb.downArrowKey.wasPressedThisFrame || kb.sKey.wasPressedThisFrame) m_HeuristicAction = 1;
        else if (kb.rightArrowKey.wasPressedThisFrame || kb.dKey.wasPressedThisFrame) m_HeuristicAction = 2;
        else if (kb.leftArrowKey.wasPressedThisFrame || kb.aKey.wasPressedThisFrame) m_HeuristicAction = 3;
#else
        if (Input.GetKeyDown(KeyCode.UpArrow) || Input.GetKeyDown(KeyCode.W)) m_HeuristicAction = 0;
        else if (Input.GetKeyDown(KeyCode.DownArrow) || Input.GetKeyDown(KeyCode.S)) m_HeuristicAction = 1;
        else if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D)) m_HeuristicAction = 2;
        else if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A)) m_HeuristicAction = 3;
#endif
    }
}
