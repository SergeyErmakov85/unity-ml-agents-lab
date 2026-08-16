using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент-охотник (TS-006): догнать цель на площадке 20×20, обходя препятствия.
///
/// Что нового по сравнению с `E04_BallBalance`:
///
/// 1. **Два сенсора вместо одного.** `RayPerceptionSensor3D` — «зрение»
///    (обход препятствий, «вижу/не вижу цель»); `VectorSensor` —
///    проприоцепция (гладкое наведение, когда цель уже видна). Один не
///    заменяет другой: по лучам нельзя точно навестись, по вектору нельзя
///    объехать стену.
/// 2. **Формирование награды (reward shaping).** Задача навигационная,
///    и «дошёл — не дошёл» на 1000 шагов — слишком редкий сигнал. Плотная
///    награда даётся в **потенциальной** форме (§ «Награда» ниже) — той
///    единственной, которая доказуемо не меняет оптимальную политику.
///
/// Важно: шагом MDP для алгоритма обучения является **решение**, а не шаг
/// физики. Поэтому `TakeActionsBetweenDecisions = false`, и `OnActionReceived`
/// вызывается ровно раз на решение — иначе формирующая награда начислялась бы
/// в пять раз чаще, чем идёт дисконтирование, и её остаточное смещение
/// перестало бы сокращаться (измерено, T-15 в docs/07_TROUBLESHOOTING.md).
///
/// Уроки курса: 2.2 (PPO), 2.4 (Reward Shaping), проект-2 «3D-охотник».
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class HunterAgent : AgentBase
{
    [Header("Ссылки")]
    public HunterArea area;

    [Header("Движение")]
    [Tooltip("Сила тяги вдоль собственного «вперёд», Н на единицу действия.")]
    public float thrustForce = 25f;

    [Tooltip("Скорость поворота, градусов в секунду на единицу действия.")]
    public float yawSpeed = 180f;

    [Tooltip("Предел линейной скорости, м/с. Он же нормировка скорости в наблюдении.")]
    public float maxSpeed = 8f;

    [Tooltip("Нормировка угловой скорости в наблюдении, рад/с.")]
    public float maxAngularSpeed = 6f;

    [Header("Награда")]
    [Tooltip("Дистанция, на которой цель считается пойманной, м.")]
    public float catchDistance = 1.0f;

    [Tooltip("Терминальная награда за поимку.")]
    public float catchReward = 1.0f;

    [Tooltip("Штраф за столкновение со стеной или препятствием.")]
    public float collisionPenalty = -0.05f;

    [Tooltip("γ формирования награды. ОБЯЗАН совпадать с gamma алгоритма обучения: " +
             "теорема Ына о неизменности оптимальной политики доказана именно для этого γ.")]
    public float shapingGamma = 0.99f;

    [Tooltip("Штраф за одно РЕШЕНИЕ. Равен 1 / (MaxStep / DecisionPeriod) = 1/200: " +
             "эпизод без результата стоит ровно -1.")]
    public float stepPenalty = -0.005f;

    [Header("Темп решений")]
    [Tooltip("Сколько шагов физики длится одно решение. Обязан совпадать с " +
             "DecisionRequester.DecisionPeriod: на него масштабируются тяга и поворот, " +
             "потому что действие применяется один раз, а действовать должно весь период.")]
    public int decisionPeriod = 5;

    public override string EnvId => "E06_Hunter3D";

    Rigidbody m_Body;
    float m_PreviousPotential;
    Vector2 m_HeuristicAction;

    /// <summary>
    /// Потенциал состояния Φ(s) = −d(s)/d_max ∈ [−1, 0].
    ///
    /// Знак минус делает потенциал тем выше, чем ближе цель, а деление
    /// на диагональ арены приводит его к безразмерному виду — иначе величина
    /// формирующей награды зависела бы от размеров площадки.
    /// </summary>
    float Potential() => -area.Distance / area.MaxDistance;

    public override void Initialize()
    {
        base.Initialize();
        m_Body = GetComponent<Rigidbody>();
        if (area == null) area = GetComponentInParent<HunterArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetEpisode(m_Body);
        m_PreviousPotential = Potential();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Все признаки нормированы в [−1, 1]. Это не косметика: без нормировки
        // разные по масштабу входы заставляют сеть тратить первые тысячи шагов
        // на выравнивание масштабов вместо решения задачи.
        Vector3 toTarget = area.target.localPosition - transform.localPosition;
        Vector3 localToTarget = transform.InverseTransformDirection(toTarget);
        Vector3 localVelocity = transform.InverseTransformDirection(m_Body.linearVelocity);

        sensor.AddObservation(localToTarget.x / area.MaxDistance);        // 1
        sensor.AddObservation(localToTarget.z / area.MaxDistance);        // 2
        sensor.AddObservation(area.Distance / area.MaxDistance);          // 3
        // Угол на цель в собственной системе координат: −1 — точно позади слева,
        // +1 — позади справа, 0 — прямо по курсу.
        sensor.AddObservation(Mathf.Atan2(localToTarget.x, localToTarget.z) / Mathf.PI);  // 4
        sensor.AddObservation(Mathf.Clamp(localVelocity.x / maxSpeed, -1f, 1f));          // 5
        sensor.AddObservation(Mathf.Clamp(localVelocity.z / maxSpeed, -1f, 1f));          // 6
        sensor.AddObservation(Mathf.Clamp(m_Body.angularVelocity.y / maxAngularSpeed, -1f, 1f));  // 7
        sensor.AddObservation(transform.forward.x);                       // 8
        sensor.AddObservation(transform.forward.z);                       // 9
        // Доля израсходованного времени: без неё среда частично наблюдаема —
        // агент не знает, сколько шагов у него осталось.
        sensor.AddObservation(Mathf.Clamp01((float)StepCount / MaxStep));  // 10
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float thrust = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float yaw = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        // Действие применяется раз в `decisionPeriod` шагов физики, поэтому
        // и тяга, и поворот масштабируются на длительность шага MDP: иначе
        // управление оказалось бы впятеро слабее, чем задумано.
        float dt = Time.fixedDeltaTime * decisionPeriod;
        transform.Rotate(Vector3.up, yaw * yawSpeed * dt);
        m_Body.AddForce(transform.forward * (thrust * thrustForce * decisionPeriod));
        if (m_Body.linearVelocity.sqrMagnitude > maxSpeed * maxSpeed)
            m_Body.linearVelocity = m_Body.linearVelocity.normalized * maxSpeed;

        bool caught = area.Distance < catchDistance;

        // Формирование награды по Ыну (1999): F(s, s′) = γ·Φ(s′) − Φ(s).
        // Потенциал терминального состояния обязан быть нулём — иначе сумма
        // формирующих наград за эпизод перестаёт телескопироваться
        // и смещает суммарную награду. В справочном коде урока этот нюанс
        // опущен; здесь он соблюдён (см. ENV_SPEC.md, §6).
        float nextPotential = caught ? 0f : Potential();
        AddReward(shapingGamma * nextPotential - m_PreviousPotential);
        m_PreviousPotential = nextPotential;

        // Штраф за шаг превращает задачу «догнать» в задачу «догнать быстрее».
        // Значение задано на ОДНО РЕШЕНИЕ, а не на шаг физики: `OnActionReceived`
        // вызывается раз в `DecisionPeriod` шагов (TakeActionsBetweenDecisions = false),
        // и именно решение является шагом MDP для алгоритма обучения.
        AddReward(stepPenalty);
        TrackStep();

        if (caught)
        {
            AddReward(catchReward);
            EndEpisodeWithResult("Catch", success: true);
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("wall") || collision.gameObject.CompareTag("obstacle"))
        {
            AddReward(collisionPenalty);
            MetricsRecorder.Sum("Collisions", 1f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuous = actionsOut.ContinuousActions;
        continuous[0] = m_HeuristicAction.x;
        continuous[1] = m_HeuristicAction.y;
    }

    void Update()
    {
        // Ввод читается в Update, а расходуется в Heuristic: FixedUpdate может
        // пропустить короткое нажатие, а Update — нет.
        float thrust = 0f;
        float yaw = 0f;
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.upArrowKey.isPressed || kb.wKey.isPressed) thrust = 1f;
        else if (kb.downArrowKey.isPressed || kb.sKey.isPressed) thrust = -1f;
        if (kb.rightArrowKey.isPressed || kb.dKey.isPressed) yaw = 1f;
        else if (kb.leftArrowKey.isPressed || kb.aKey.isPressed) yaw = -1f;
#else
        if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) thrust = 1f;
        else if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) thrust = -1f;
        if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) yaw = 1f;
        else if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) yaw = -1f;
#endif
        m_HeuristicAction = new Vector2(thrust, yaw);
    }
}
