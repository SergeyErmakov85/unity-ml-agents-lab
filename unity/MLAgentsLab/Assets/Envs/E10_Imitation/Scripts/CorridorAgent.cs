using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент среды `E10_Imitation` (TS-010): пройти фиксированный змеевидный
/// коридор длиной 48 ходов при лимите 120.
///
/// Главное требование к наблюдению этой среды: **скриптовый эксперт обязан
/// уметь действовать только по нему**. Иначе демонстрации содержали бы
/// знание, недоступное агенту, и имитационное обучение училось бы
/// невозможному — политика идеально повторяла бы эксперта на записях
/// и разваливалась в среде.
///
/// Отсюда два признака, которых нет в других средах лаборатории:
///
/// * индексы 7–10 — «щупальца»: есть ли стена в каждой из четырёх соседних
///   клеток. Их достаточно, чтобы пройти коридор без развилок правилом
///   «не возвращайся назад, иди куда можно»;
/// * индекс 12 — направление предыдущего хода. Без него правило
///   «не возвращайся» невыразимо: в клетке коридора два прохода, и какой
///   из них «назад», знает только история.
///
/// Лучевого сенсора здесь нет: он дал бы ту же информацию расплывчато
/// и заставил бы эксперта интерпретировать длины лучей.
///
/// Урок курса: 3.4 «Imitation Learning: Behavioral Cloning и GAIL».
/// </summary>
public class CorridorAgent : AgentBase
{
    [Header("Ссылки")]
    public CorridorArea area;

    [Header("Награда")]
    [Tooltip("Награда за достижение выхода.")]
    public float goalReward = 1f;

    [Tooltip("Штраф за ход в стену. Агент остаётся на месте.")]
    public float wallPenalty = -0.02f;

    [Tooltip("Штраф за шаг. Равен −1/MaxStep: эпизод без результата стоит ровно −1.")]
    public float stepPenalty = -1f / 120f;

    public override string EnvId => "E10_Imitation";

    /// <summary>Не дошёл за отведённое время — это неудача.</summary>
    protected override bool TimeoutIsSuccess => false;

    /// <summary>Направление предыдущего хода; −1 в начале эпизода.</summary>
    int m_PreviousAction = -1;

    int m_HeuristicAction = -1;

    public override void Initialize()
    {
        base.Initialize();
        if (area == null) area = GetComponentInParent<CorridorArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetEpisode();
        m_PreviousAction = -1;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        const float scale = CorridorArea.GridSize - 1;
        var cell = area.AgentCell;
        var goal = CorridorArea.GoalCell;

        sensor.AddObservation(cell.x / scale);                              // 0
        sensor.AddObservation(cell.y / scale);                             // 1
        sensor.AddObservation(goal.x / scale);                             // 2
        sensor.AddObservation(goal.y / scale);                             // 3
        sensor.AddObservation((goal.x - cell.x) / scale);                  // 4
        sensor.AddObservation((goal.y - cell.y) / scale);                  // 5
        sensor.AddObservation((Mathf.Abs(goal.x - cell.x)
                             + Mathf.Abs(goal.y - cell.y)) / (2f * scale));  // 6

        // «Щупальца»: порядок обязан совпадать с порядком действий
        // (север, юг, восток, запад) — эксперт индексирует их напрямую.
        sensor.AddObservation(area.IsWall(cell.x, cell.y + 1) ? 1f : 0f);  // 7  север
        sensor.AddObservation(area.IsWall(cell.x, cell.y - 1) ? 1f : 0f);  // 8  юг
        sensor.AddObservation(area.IsWall(cell.x + 1, cell.y) ? 1f : 0f);  // 9  восток
        sensor.AddObservation(area.IsWall(cell.x - 1, cell.y) ? 1f : 0f);  // 10 запад

        sensor.AddObservation(MaxStep > 0 ? Mathf.Clamp01((float)StepCount / MaxStep) : 0f);  // 11
        // −1/3 в начале эпизода: значение вне диапазона реальных действий,
        // чтобы «хода ещё не было» отличалось от «ходил на север».
        sensor.AddObservation(m_PreviousAction / 3f);                      // 12
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int direction = actions.DiscreteActions[0];

        if (!area.TryMove(direction))
        {
            AddReward(wallPenalty);
            MetricsRecorder.Sum("WallBumps", 1f);
        }

        m_PreviousAction = direction;
        AddReward(stepPenalty);
        TrackStep();

        if (area.AtGoal)
        {
            AddReward(goalReward);
            MetricsRecorder.Average("Progress", 1f);
            EndEpisodeWithResult("Goal", success: true);
        }
        else if (MaxStep > 0 && StepCount >= MaxStep - 1)
        {
            // Прогресс на момент обрыва — главная диагностика среды: доля
            // успехов у RL долго равна нулю, и без неё непонятно, идёт ли
            // обучение вообще. Записывается на ПОСЛЕДНЕМ шаге: после обрыва
            // OnEpisodeBegin уже вернёт агента на старт.
            MetricsRecorder.Average("Progress", area.Progress);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        // −1 означает «клавиша не нажата»: повторять последнее направление
        // нельзя, иначе агент едет сам и проверить среду руками не выйдет.
        discrete[0] = m_HeuristicAction >= 0 ? m_HeuristicAction : 2;
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
