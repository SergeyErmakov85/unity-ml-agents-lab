using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент среды `E09_CurriculumMaze` (TS-009): пройти лабиринт от старта к выходу.
///
/// Что нового по сравнению с остальными средами лаборатории:
///
/// 1. **Задача меняется по ходу обучения.** Размер сетки и плотность стен
///    зависят от `difficulty`, которую Python двигает учебным планом.
///    Поэтому текущая сложность и размер сетки **входят в наблюдение**:
///    без них одно и то же «я в клетке (2,2), выход в (4,4)» означает разное
///    на карте 5 × 5 и на карте 11 × 11.
///
/// 2. **Формирования награды нет намеренно.** Потенциал по расстоянию до
///    выхода в лабиринте вреден: он тянет агента к стене, за которой цель,
///    а обходной путь ведёт «от» цели и штрафуется. Роль плотного сигнала
///    здесь играет учебный план — на простом уровне выход находится
///    случайно, и дальше сигнал уже есть. Это и есть ответ на вопрос,
///    зачем нужен curriculum, когда есть reward shaping (`ENV_SPEC.md`, §6).
///
/// 3. **Движение телепортом по клеткам**, как в `E01_GridWorld`: без
///    `Rigidbody`. Физика здесь ничего не добавляет к задаче, а недетерминизм
///    добавляет.
///
/// Урок курса: 3.3 «Учебный план и рандомизация среды».
/// </summary>
public class MazeAgent : AgentBase
{
    [Header("Ссылки")]
    public MazeArea area;

    [Header("Награда")]
    [Tooltip("Награда за достижение выхода.")]
    public float goalReward = 1f;

    [Tooltip("Штраф за ход в стену. Агент остаётся на месте.")]
    public float wallPenalty = -0.02f;

    [Tooltip("Штраф за шаг. Равен −1/MaxStep: эпизод без результата стоит ровно −1.")]
    public float stepPenalty = -1f / 150f;

    public override string EnvId => "E09_CurriculumMaze";

    /// <summary>Не дошёл до выхода за отведённое время — это неудача.</summary>
    protected override bool TimeoutIsSuccess => false;

    int m_HeuristicAction = -1;

    public override void Initialize()
    {
        base.Initialize();
        if (area == null) area = GetComponentInParent<MazeArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetEpisode();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Позиции даны в КЛЕТКАХ, а не в метрах: задача дискретна, и перевод
        // в метры заставил бы сеть переводить обратно.
        float scale = Mathf.Max(1, area.GridSize - 1);

        sensor.AddObservation(area.AgentCell.x / scale);                        // 0
        sensor.AddObservation(area.AgentCell.y / scale);                        // 1
        sensor.AddObservation(area.GoalCell.x / scale);                         // 2
        sensor.AddObservation(area.GoalCell.y / scale);                         // 3
        sensor.AddObservation((area.GoalCell.x - area.AgentCell.x) / scale);    // 4
        sensor.AddObservation((area.GoalCell.y - area.AgentCell.y) / scale);    // 5

        int manhattan = Mathf.Abs(area.GoalCell.x - area.AgentCell.x)
                      + Mathf.Abs(area.GoalCell.y - area.AgentCell.y);
        sensor.AddObservation(manhattan / (2f * scale));                        // 6

        sensor.AddObservation(MaxStep > 0 ? Mathf.Clamp01((float)StepCount / MaxStep) : 0f);  // 7

        // Два признака сложности — не техническая деталь, а часть задачи:
        // сетка меняется между эпизодами, и без них среда становится
        // частично наблюдаемой по устранимой причине (ENV_SPEC.md, §4).
        sensor.AddObservation(Mathf.Clamp01(area.Difficulty));                  // 8
        sensor.AddObservation(
            (area.GridSize - MazeArea.MinGridSize) /
            (float)(MazeArea.MaxGridSize - MazeArea.MinGridSize));              // 9
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int direction = actions.DiscreteActions[0];

        // Ход в стену не запрещён маской, а штрафуется: маска сделала бы
        // задачу проще ценой того самого умения, которому среда учит, —
        // читать лучи и понимать, где стена (ENV_SPEC.md, §5).
        if (!area.TryMove(direction))
        {
            AddReward(wallPenalty);
            MetricsRecorder.Sum("WallBumps", 1f);
        }

        AddReward(stepPenalty);
        TrackStep();

        if (area.AtGoal)
        {
            AddReward(goalReward);
            EndEpisodeWithResult("Goal", success: true);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        // −1 означает «клавиша не нажата»: повторять последнее направление
        // нельзя, иначе агент едет сам по себе и проверить среду руками
        // не получается. Действие 0 (север) — нейтральный выбор по умолчанию.
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
