using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент-шар, который учится докатываться до цели, не падая с платформы (TS-003).
///
/// Наблюдения (8): локальные позиции цели и агента по 3 и скорость агента по X, Z.
/// Позиции **локальные** относительно арены — иначе восемь арен, разнесённых
/// по X, выглядели бы для сети восемью разными задачами.
///
/// Действия: Discrete 1 ветвь × 4 — направление прикладываемой силы.
/// Непрерывный вариант (Continuous 2) остаётся для PPO/SAC в Фазе 3; переход
/// затрагивает только <see cref="OnActionReceived"/> и `ActionSpec` в Setup-скрипте.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class RollerAgent : AgentBase
{
    [Header("Ссылки")]
    public RollerArea area;

    [Header("Параметры движения")]
    [Tooltip("Множитель силы, прикладываемой к Rigidbody за шаг.")]
    public float forceMultiplier = 10f;

    [Header("Награды")]
    [Tooltip("Штраф за каждый шаг Academy. Равен 1/MaxStep: эпизод без результата стоит ровно -1.")]
    public float stepReward = -0.001f;
    public float goalReward = 1f;
    public float fallReward = -1f;

    [Tooltip("Дистанция, на которой цель считается достигнутой.")]
    public float reachDistance = 1.42f;

    public override string EnvId => "E03_RollerBall";

    Rigidbody m_Body;
    int m_HeuristicAction;

    /// <summary>Направления силы по индексу действия. Порядок зафиксирован в TS-003, §5.</summary>
    static readonly Vector3[] Directions =
    {
        Vector3.forward,  // 0: +Z
        Vector3.back,     // 1: -Z
        Vector3.left,     // 2: -X
        Vector3.right,    // 3: +X
    };

    public override void Initialize()
    {
        base.Initialize();
        m_Body = GetComponent<Rigidbody>();
        if (area == null) area = GetComponentInParent<RollerArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetAgent(m_Body);
        area.PlaceTarget();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(area.target.localPosition);  // 3
        sensor.AddObservation(transform.localPosition);    // 3
        sensor.AddObservation(m_Body.linearVelocity.x);    // 1
        sensor.AddObservation(m_Body.linearVelocity.z);    // 1
        // Итого 8 — совпадает со строкой-контрактом ENV_SPEC.md.
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int action = actions.DiscreteActions[0];
        m_Body.AddForce(Directions[action] * forceMultiplier);

        AddReward(stepReward);
        // Обрыв по MaxStep выполняет ML-Agents раньше следующего OnActionReceived,
        // поэтому награда и длина эпизода запоминаются на каждом шаге.
        TrackStep();

        float distance = Vector3.Distance(transform.localPosition, area.target.localPosition);
        if (distance < reachDistance)
        {
            AddReward(goalReward);
            EndEpisodeWithResult("Goal", success: true);
        }
        else if (area.HasFallen())
        {
            AddReward(fallReward);
            EndEpisodeWithResult("Fall", success: false);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // DiscreteActions — свойство, возвращающее структуру-сегмент, поэтому
        // индексировать надо локальную копию, а не результат свойства напрямую
        // (иначе CS1612: изменение возвращаемого значения).
        var discrete = actionsOut.DiscreteActions;
        discrete[0] = m_HeuristicAction;
    }

    void Update()
    {
        // Ввод читается в Update, а расходуется в Heuristic: FixedUpdate может
        // пропустить короткое нажатие, а Update — нет.
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.upArrowKey.isPressed || kb.wKey.isPressed) m_HeuristicAction = 0;
        else if (kb.downArrowKey.isPressed || kb.sKey.isPressed) m_HeuristicAction = 1;
        else if (kb.leftArrowKey.isPressed || kb.aKey.isPressed) m_HeuristicAction = 2;
        else if (kb.rightArrowKey.isPressed || kb.dKey.isPressed) m_HeuristicAction = 3;
#else
        if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) m_HeuristicAction = 0;
        else if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) m_HeuristicAction = 1;
        else if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) m_HeuristicAction = 2;
        else if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) m_HeuristicAction = 3;
#endif
    }
}
