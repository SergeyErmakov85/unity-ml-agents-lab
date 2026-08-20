using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Игрок среды `E08_SoccerArena` (TS-008).
///
/// Что здесь принципиально нового по сравнению с одноагентными средами:
///
/// 1. **Наблюдение перспективно-верное.** Одна и та же сеть управляет обеими
///    командами, поэтому вход обязан значить одно и то же с обеих сторон.
///    Для команды East мир зеркалится по X и Z (<see cref="SoccerArea.MirrorSign"/>),
///    а порядок тегов лучевого сенсора переставляется так, чтобы позиция 4
///    всегда означала «чужие ворота» (ТЗ §4). Без этого разделение весов между
///    командами — обязательное для self-play — стало бы источником
///    неустранимой ошибки: сеть училась бы двум противоположным задачам под
///    одним входом.
///
/// 2. **Награду за гол начисляет не агент, а арена** — через
///    <c>SimpleMultiAgentGroup</c>. Личной у игрока остаётся только плата
///    за время: это единственное, что он контролирует единолично.
///
/// 3. **Эпизод завершает арена сразу у всех четверых.** Агент лишь сообщает
///    арене об истечении <c>MaxStep</c>.
///
/// Урок курса: 3.2 (MA-POCA и Self-Play).
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class SoccerPlayerAgent : AgentBase
{
    [Header("Роль")]
    public SoccerArea area;

    [Tooltip("Сторона поля. Обязана совпадать с TeamId в BehaviorParameters.")]
    public SoccerArea.Side side = SoccerArea.Side.West;

    [Tooltip("Управляется ли этот игрок с клавиатуры в режиме Heuristic. " +
             "Ровно один игрок в сцене — иначе одна клавиша двигала бы всех.")]
    public bool keyboardControlled;

    [Tooltip("Начисляет ли этот игрок командную формирующую награду за продвижение " +
             "мяча. Ровно один игрок в каждой команде: награда командная, и начислить " +
             "её от каждого значило бы умножить на размер команды.")]
    public bool shapingReporter;

    [Header("Движение")]
    [Tooltip("Сила тяги вдоль собственных осей, Н на единицу действия.")]
    public float thrustForce = 30f;

    [Tooltip("Доля тяги, доступная при движении боком: перемещаться вбок труднее.")]
    [Range(0f, 1f)]
    public float strafeFactor = 0.6f;

    [Tooltip("Скорость поворота, градусов в секунду.")]
    public float yawSpeed = 180f;

    [Tooltip("Предел скорости, м/с. Он же нормировка скорости в наблюдении.")]
    public float maxSpeed = 8f;

    [Header("Награда")]
    [Tooltip("Личный штраф за одно РЕШЕНИЕ. Значение −0.5/600 даёт ровно −0.5 " +
             "за эпизод, прожитый без гола: вдвое дешевле пропущенного мяча.")]
    public float stepPenalty = -0.000833f;

    [Header("Темп решений")]
    [Tooltip("Сколько шагов физики длится одно решение. Обязан совпадать с " +
             "DecisionRequester.DecisionPeriod: на него масштабируется тяга.")]
    public int decisionPeriod = 5;

    public override string EnvId => "E08_SoccerArena";

    /// <summary>
    /// Обрыв по времени в футболе — ничья, а не поражение, но и не успех:
    /// метрика `Env/SuccessRate` считает долей успеха именно забитый гол.
    /// </summary>
    protected override bool TimeoutIsSuccess => false;

    Rigidbody m_Body;
    Vector3Int m_HeuristicAction;
    bool m_TimeoutReported;

    /// <summary>Знак зеркалирования мира для этой стороны: West +1, East −1.</summary>
    float Sign => SoccerArea.MirrorSign(side);

    public override void Initialize()
    {
        base.Initialize();
        m_Body = GetComponent<Rigidbody>();
        if (area == null) area = GetComponentInParent<SoccerArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        m_TimeoutReported = false;
        // Расстановкой занимается арена: она обязана поставить всех четверых
        // согласованно, а вызывается OnEpisodeBegin у каждого по отдельности.
    }

    /// <summary>
    /// Фиксирует исход матча для отладочного HUD и наблюдателя инференса.
    /// Вызывается ареной до <c>EndGroupEpisode</c>.
    /// </summary>
    public void NoteMatchResult(string result)
    {
        LastEpisodeResult = result;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Всё — в командной системе координат: для East оси X и Z инвертированы,
        // поэтому «к чужим воротам» для обеих команд означает +x.
        float sign = Sign;

        Vector3 self = transform.localPosition;
        Vector3 ball = area.BallPosition;
        Vector3 velocity = m_Body.linearVelocity;

        sensor.AddObservation(Mathf.Clamp(sign * self.x / area.halfLength, -1f, 1f));            // 0
        sensor.AddObservation(Mathf.Clamp(sign * self.z / area.halfWidth, -1f, 1f));             // 1
        sensor.AddObservation(Mathf.Clamp(sign * (ball.x - self.x) / area.halfLength, -1f, 1f)); // 2
        sensor.AddObservation(Mathf.Clamp(sign * (ball.z - self.z) / area.halfWidth, -1f, 1f));  // 3
        sensor.AddObservation(Mathf.Clamp(sign * velocity.x / maxSpeed, -1f, 1f));               // 4
        sensor.AddObservation(Mathf.Clamp(sign * velocity.z / maxSpeed, -1f, 1f));               // 5
        // Косинус курса в командной системе: +1 — смотрю на чужие ворота.
        sensor.AddObservation(sign * transform.forward.x);                                       // 6
        // Доля израсходованного времени: без неё среда частично наблюдаема —
        // агент не знает, сколько решений у него осталось, а цена гола падает
        // со временем (ТЗ §6).
        sensor.AddObservation(MaxStep > 0 ? Mathf.Clamp01((float)StepCount / MaxStep) : 0f);     // 7
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int forward = actions.DiscreteActions[0];   // 0 стоп, 1 вперёд, 2 назад
        int strafe = actions.DiscreteActions[1];    // 0 стоп, 1 вправо, 2 влево
        int rotate = actions.DiscreteActions[2];    // 0 стоп, 1 по часовой, 2 против

        float dt = Time.fixedDeltaTime * decisionPeriod;

        float longitudinal = forward == 1 ? 1f : forward == 2 ? -1f : 0f;
        float lateral = strafe == 1 ? 1f : strafe == 2 ? -1f : 0f;
        float yaw = rotate == 1 ? 1f : rotate == 2 ? -1f : 0f;

        transform.Rotate(Vector3.up, yaw * yawSpeed * dt);

        // Тяга в СОБСТВЕННЫХ осях игрока, а не в осях поля: зеркалирование
        // касается только наблюдения. Мир один, и физика в нём одна.
        Vector3 force = transform.forward * (longitudinal * thrustForce)
                      + transform.right * (lateral * thrustForce * strafeFactor);
        m_Body.AddForce(force * decisionPeriod);

        if (m_Body.linearVelocity.sqrMagnitude > maxSpeed * maxSpeed)
            m_Body.linearVelocity = m_Body.linearVelocity.normalized * maxSpeed;

        AddReward(stepPenalty);

        // Командная формирующая награда начисляется ровно один раз за шаг MDP:
        // все игроки решают на одном и том же шаге, поэтому «докладчиком»
        // назначен один из них (ENV_SPEC.md, §6).
        if (shapingReporter) area.ApplyBallShaping(side);

        TrackStep();

        // Обрыв по времени: ML-Agents завершит эпизод каждого агента по MaxStep
        // сам, но группа об этом не узнает, и Python получит терминалы без
        // группового шага. Сообщаем арене на последнем решении.
        if (!m_TimeoutReported && MaxStep > 0 && StepCount >= MaxStep - decisionPeriod)
        {
            m_TimeoutReported = true;
            area.RegisterTimeout();
        }
    }

    /// <summary>
    /// Касание мяча. Награды не даёт: арена запоминает сторону последнего
    /// касания, чтобы отличить автогол от обычного, и считает метрику.
    /// </summary>
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("ball"))
            area.NoteBallTouch(side);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        discrete[0] = m_HeuristicAction.x;
        discrete[1] = m_HeuristicAction.y;
        discrete[2] = m_HeuristicAction.z;
    }

    void Update()
    {
        if (!keyboardControlled) return;

        // Ввод читается в Update, а расходуется в Heuristic: FixedUpdate
        // может пропустить короткое нажатие, а Update — нет.
        int forward = 0, strafe = 0, rotate = 0;
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.wKey.isPressed) forward = 1; else if (kb.sKey.isPressed) forward = 2;
        if (kb.dKey.isPressed) strafe = 1; else if (kb.aKey.isPressed) strafe = 2;
        if (kb.eKey.isPressed) rotate = 1; else if (kb.qKey.isPressed) rotate = 2;
#else
        if (Input.GetKey(KeyCode.W)) forward = 1; else if (Input.GetKey(KeyCode.S)) forward = 2;
        if (Input.GetKey(KeyCode.D)) strafe = 1; else if (Input.GetKey(KeyCode.A)) strafe = 2;
        if (Input.GetKey(KeyCode.E)) rotate = 1; else if (Input.GetKey(KeyCode.Q)) rotate = 2;
#endif
        m_HeuristicAction = new Vector3Int(forward, strafe, rotate);
    }
}
