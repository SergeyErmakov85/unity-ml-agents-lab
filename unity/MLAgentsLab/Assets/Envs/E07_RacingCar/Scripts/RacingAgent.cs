using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент-гонщик (TS-007): проехать круг по замкнутому треку, не врезаясь
/// в стены. Проект-3 курса, урок 3.1 (SAC).
///
/// Почему именно эта среда выбрана для SAC. Шаг здесь дорог: эпизод длится
/// до 400 решений, а неудачный заезд заканчивается столкновением через
/// десяток шагов. SAC — off-policy, он переиспользует каждый переход
/// многократно и потому доходит до результата за меньшее число шагов среды,
/// чем PPO. Это и есть его практическая ниша (урок 3.1, «SAC vs PPO»).
///
/// Наблюдение снова из двух источников: 9 лучей веером на 180° («вижу стену»)
/// и 8 чисел проприоцепции («как еду и куда следующая точка»).
///
/// Модель движения кинематическая: действие задаёт скорость и курс напрямую.
/// Инерция и демпфирование к предмету урока не относятся, а управляемость
/// портят заметно (ENV_SPEC.md, §15).
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class RacingAgent : AgentBase
{
    [Header("Ссылки")]
    public RacingArea area;

    [Header("Управление")]
    [Tooltip("Скорость поворота, градусов в секунду на единицу действия.")]
    public float turnSpeed = 150f;

    [Tooltip("Максимальная скорость, м/с. Она же нормировка скорости в наблюдении.")]
    public float maxSpeed = 12f;

    [Tooltip("Минимальная скорость, м/с: машина не умеет останавливаться совсем. " +
             "Без этого выгодной стратегией становится «стоять и не рисковать».")]
    public float minSpeed = 2f;

    [Tooltip("Нормировка угловой скорости в наблюдении, рад/с.")]
    public float maxAngularSpeed = 6f;

    [Tooltip("Сколько шагов физики длится одно решение. Обязан совпадать с " +
             "DecisionRequester.DecisionPeriod: на него масштабируются газ и руль.")]
    public int decisionPeriod = 5;

    [Header("Награда")]
    [Tooltip("Награда за скорость, на решение при полной скорости. " +
             "Мала намеренно: см. ENV_SPEC.md, §6 — при большом значении выгоднее " +
             "быстро наматывать круги мимо контрольных точек, чем проходить их.")]
    public float speedReward = 0.002f;

    [Tooltip("Штраф за решение: превращает «проехать круг» в «проехать круг быстро».")]
    public float stepPenalty = -0.001f;

    [Tooltip("Награда за очередную контрольную точку, взятую по порядку.")]
    public float checkpointReward = 1.0f;

    [Tooltip("Награда за полный круг. Выдаётся сверх награды за последнюю точку.")]
    public float lapReward = 5.0f;

    [Tooltip("Итоговая награда эпизода при столкновении со стеной. " +
             "Задаётся через SetReward: авария обесценивает заезд целиком.")]
    public float crashReward = -1.0f;

    public override string EnvId => "E07_RacingCar";

    Rigidbody m_Body;
    Vector2 m_HeuristicAction;

    public override void Initialize()
    {
        base.Initialize();
        m_Body = GetComponent<Rigidbody>();
        if (area == null) area = GetComponentInParent<RacingArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetEpisode(m_Body);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 localVelocity = transform.InverseTransformDirection(m_Body.linearVelocity);
        Vector3 toCheckpoint = area.NextCheckpointPosition - transform.localPosition;
        Vector3 localToCheckpoint = transform.InverseTransformDirection(toCheckpoint);

        sensor.AddObservation(Mathf.Clamp(m_Body.linearVelocity.magnitude / maxSpeed, 0f, 1f));   // 1
        sensor.AddObservation(Mathf.Clamp(localVelocity.x / maxSpeed, -1f, 1f));                  // 2
        sensor.AddObservation(Mathf.Clamp(localVelocity.z / maxSpeed, -1f, 1f));                  // 3
        sensor.AddObservation(Mathf.Clamp(m_Body.angularVelocity.y / maxAngularSpeed, -1f, 1f));  // 4
        // Направление на следующую точку в системе координат машины —
        // нормированный вектор, а не разность координат: важно «куда рулить»,
        // а не «как далеко», для расстояния есть отдельный признак.
        Vector3 direction = localToCheckpoint.sqrMagnitude > 1e-6f
            ? localToCheckpoint.normalized
            : Vector3.forward;
        sensor.AddObservation(direction.x);                                                      // 5
        sensor.AddObservation(direction.z);                                                      // 6
        sensor.AddObservation(Mathf.Clamp01(localToCheckpoint.magnitude / area.LapLength));       // 7
        sensor.AddObservation(area.Progress);                                                    // 8
        // Итого 8 — совпадает со строкой-контрактом ENV_SPEC.md.
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float throttle = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float steer = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        // Кинематическая модель: действие задаёт скорость и курс напрямую,
        // а не силу. Так сделано сознательно (ENV_SPEC.md, §15): модель через
        // AddForce добавляет к задаче инерцию и демпфирование, которые
        // к предмету урока об SAC отношения не имеют, зато делают управление
        // непредсказуемым — измерено, машина не проходила круг даже под
        // простым пропорциональным регулятором.
        float dt = Time.fixedDeltaTime * decisionPeriod;
        transform.Rotate(0f, steer * turnSpeed * dt, 0f);

        float targetSpeed = Mathf.Lerp(minSpeed, maxSpeed, 0.5f * (throttle + 1f));
        m_Body.linearVelocity = transform.forward * targetSpeed;

        float speed = targetSpeed / maxSpeed;
        AddReward(speedReward * speed);
        AddReward(stepPenalty);
        TrackStep();
    }

    void OnTriggerEnter(Collider other)
    {
        if (!other.CompareTag("goal")) return;
        if (!area.TryPassCheckpoint(other.transform)) return;

        AddReward(checkpointReward);
        MetricsRecorder.Sum("Checkpoints", 1f);

        if (area.LapCompleted)
        {
            AddReward(lapReward);
            TrackStep();
            EndEpisodeWithResult("Lap", success: true);
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (!collision.gameObject.CompareTag("wall")) return;

        // SetReward, а не AddReward: авария обесценивает весь заезд целиком,
        // и итоговая награда эпизода становится ровно −1 независимо от того,
        // сколько точек агент успел взять. Так «доехать до третьей точки
        // и разбиться» перестаёт быть выгоднее, чем «ехать осторожно».
        SetReward(crashReward);
        MetricsRecorder.Sum("Crashes", 1f);
        TrackStep();
        EndEpisodeWithResult("Crash", success: false);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuous = actionsOut.ContinuousActions;
        continuous[0] = m_HeuristicAction.x;
        continuous[1] = m_HeuristicAction.y;
    }

    void Update()
    {
        float accel = 0f;
        float steer = 0f;
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.upArrowKey.isPressed || kb.wKey.isPressed) accel = 1f;
        else if (kb.downArrowKey.isPressed || kb.sKey.isPressed) accel = -1f;
        if (kb.rightArrowKey.isPressed || kb.dKey.isPressed) steer = 1f;
        else if (kb.leftArrowKey.isPressed || kb.aKey.isPressed) steer = -1f;
#else
        if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) accel = 1f;
        else if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) accel = -1f;
        if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) steer = 1f;
        else if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) steer = -1f;
#endif
        m_HeuristicAction = new Vector2(accel, steer);
    }
}
