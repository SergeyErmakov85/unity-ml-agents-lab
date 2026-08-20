using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент-собиратель (TS-005): собрать как можно больше хорошей еды и не
/// хватать плохую. Флагманский пример курса, урок 2.1 (REINFORCE).
///
/// Две новые для лаборатории вещи:
///
/// 1. **Сеточное наблюдение (`GridSensor`).** Агент видит не список объектов,
///    а карту вокруг себя: тензор (2, 8, 8) — по каналу на сорт еды. Это
///    первое в лаборатории наблюдение, которое нельзя приклеить к вектору:
///    оно уходит в политику отдельным входом `obs_0` и обрабатывается
///    свёрточной частью сети.
/// 2. **Гибридное пространство действий.** Движение непрерывно (тяга,
///    поворот), а ускорение — дискретный переключатель. Контракт ONNX
///    поддерживает обе группы выходов одновременно; политика обязана
///    возвращать и среднее непрерывной части, и логиты дискретной.
///
/// Ускорение — не украшение, а осмысленный выбор: оно втрое увеличивает
/// тягу, но стоит награды каждый шаг. Агент должен научиться включать его,
/// когда до еды далеко, и выключать, когда он уже среди неё.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class FoodCollectorAgent : AgentBase
{
    [Header("Ссылки")]
    public FoodCollectorArea area;

    [Header("Движение")]
    [Tooltip("Сила тяги, Н на единицу действия.")]
    public float thrustForce = 20f;

    [Tooltip("Скорость поворота, градусов в секунду на единицу действия.")]
    public float yawSpeed = 180f;

    [Tooltip("Предел скорости без ускорения, м/с. Он же нормировка в наблюдении.")]
    public float maxSpeed = 6f;

    [Tooltip("Во сколько раз ускорение увеличивает тягу и предел скорости.")]
    public float boostMultiplier = 3f;

    [Tooltip("Нормировка угловой скорости в наблюдении, рад/с.")]
    public float maxAngularSpeed = 6f;

    [Tooltip("Сколько шагов физики длится одно решение. Обязан совпадать с " +
             "DecisionRequester.DecisionPeriod: на него масштабируются тяга и поворот.")]
    public int decisionPeriod = 5;

    [Header("Награда")]
    [Tooltip("Награда за хорошую еду.")]
    public float goodFoodReward = 1f;

    [Tooltip("Штраф за плохую еду.")]
    public float badFoodReward = -1f;

    [Tooltip("Штраф за решение: без него выгодно стоять на месте, ничего не трогая.")]
    public float stepPenalty = -0.001f;

    [Tooltip("Дополнительный штраф за решение при включённом ускорении.")]
    public float boostPenalty = -0.002f;

    public override string EnvId => "E05_FoodCollector";

    /// <summary>Сбор еды — задача без провала: эпизод всегда доживает до конца.</summary>
    protected override bool TimeoutIsSuccess => true;

    Rigidbody m_Body;
    bool m_BoostActive;
    Vector2 m_HeuristicMove;
    int m_HeuristicBoost;

    public override void Initialize()
    {
        base.Initialize();
        m_Body = GetComponent<Rigidbody>();
        if (area == null) area = GetComponentInParent<FoodCollectorArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetEpisode(m_Body);
        m_BoostActive = false;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Карта еды приходит отдельным сенсором (GridSensor). Здесь —
        // только проприоцепция: где еда, агент видит, а как он сам движется —
        // нет.
        Vector3 localVelocity = transform.InverseTransformDirection(m_Body.linearVelocity);

        sensor.AddObservation(Mathf.Clamp(localVelocity.x / maxSpeed, -1f, 1f));                  // 1
        sensor.AddObservation(Mathf.Clamp(localVelocity.z / maxSpeed, -1f, 1f));                  // 2
        sensor.AddObservation(Mathf.Clamp(m_Body.angularVelocity.y / maxAngularSpeed, -1f, 1f));  // 3
        sensor.AddObservation(transform.forward.x);                                              // 4
        sensor.AddObservation(transform.forward.z);                                              // 5
        sensor.AddObservation(Mathf.Clamp01((float)StepCount / MaxStep));                        // 6
        // Итого 6 — совпадает со строкой-контрактом ENV_SPEC.md.
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float thrust = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float yaw = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        m_BoostActive = actions.DiscreteActions[0] == 1;

        float multiplier = m_BoostActive ? boostMultiplier : 1f;
        float dt = Time.fixedDeltaTime * decisionPeriod;

        transform.Rotate(Vector3.up, yaw * yawSpeed * dt);
        m_Body.AddForce(transform.forward * (thrust * thrustForce * multiplier * decisionPeriod));

        float speedLimit = maxSpeed * multiplier;
        if (m_Body.linearVelocity.sqrMagnitude > speedLimit * speedLimit)
            m_Body.linearVelocity = m_Body.linearVelocity.normalized * speedLimit;

        AddReward(stepPenalty);
        if (m_BoostActive) AddReward(boostPenalty);
        TrackStep();
    }

    void OnTriggerEnter(Collider other)
    {
        bool good = other.CompareTag("goal");
        if (!good && !other.CompareTag("trap")) return;

        AddReward(good ? goodFoodReward : badFoodReward);
        MetricsRecorder.Sum(good ? "GoodFood" : "BadFood", 1f);
        // Еда переставляется, а не исчезает: иначе её плотность падает
        // по ходу эпизода и задача меняется на середине.
        area.RespawnFood(other.transform);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuous = actionsOut.ContinuousActions;
        continuous[0] = m_HeuristicMove.x;
        continuous[1] = m_HeuristicMove.y;

        var discrete = actionsOut.DiscreteActions;
        discrete[0] = m_HeuristicBoost;
    }

    void Update()
    {
        float thrust = 0f;
        float yaw = 0f;
        int boost = 0;
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.upArrowKey.isPressed || kb.wKey.isPressed) thrust = 1f;
        else if (kb.downArrowKey.isPressed || kb.sKey.isPressed) thrust = -1f;
        if (kb.rightArrowKey.isPressed || kb.dKey.isPressed) yaw = 1f;
        else if (kb.leftArrowKey.isPressed || kb.aKey.isPressed) yaw = -1f;
        if (kb.spaceKey.isPressed) boost = 1;
#else
        if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) thrust = 1f;
        else if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) thrust = -1f;
        if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) yaw = 1f;
        else if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) yaw = -1f;
        if (Input.GetKey(KeyCode.Space)) boost = 1;
#endif
        m_HeuristicMove = new Vector2(thrust, yaw);
        m_HeuristicBoost = boost;
    }
}
