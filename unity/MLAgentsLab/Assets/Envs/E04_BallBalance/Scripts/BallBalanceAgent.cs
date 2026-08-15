using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент-платформа, удерживающая шар (TS-004). Компонент висит на самой
/// платформе: действие агента — это её наклон.
///
/// Учебный смысл: **первая среда с непрерывными действиями**. Дискретный набор
/// «влево/вправо» здесь принципиально недостаточен — нужен угол, а не
/// направление. Это переводит задачу из семейства value-based (`E03`)
/// в семейство policy gradient: политика выдаёт параметры распределения
/// (среднее гауссианы), а не значения действий. Уроки 2.3 (A2C, гауссова
/// политика) и проект-1 «Баланс в 3D» (PPO).
///
/// Про диапазон действий. Действие приходит из сети в диапазоне [−1, 1]:
/// экспортёр ONNX приводит выход политики к нему по контракту ML-Agents
/// (`clamp(x, −3, 3) / 3`, см. `docs/04_ONNX_CONTRACT.md`, §4.2), и точно то же
/// преобразование применяется на стороне Python при сборе опыта. Здесь действие
/// дополнительно ограничивается — на случай политики, обученной чужим кодом.
/// </summary>
public class BallBalanceAgent : AgentBase
{
    [Header("Ссылки")]
    public BallBalanceArea area;

    [Header("Управление")]
    [Tooltip("Максимальный поворот платформы за один шаг, градусов на единицу действия.")]
    public float rotationSpeed = 2f;

    [Tooltip("Предельный наклон платформы в единицах компоненты кватерниона (0.25 ≈ 29°).")]
    public float tiltLimit = 0.25f;

    [Header("Награды")]
    [Tooltip("Награда за каждый шаг, на котором шар ещё на платформе.")]
    public float stepReward = 0.1f;

    [Tooltip("Штраф за падение шара.")]
    public float dropReward = -1f;

    public override string EnvId => "E04_BallBalance";

    /// <summary>
    /// Дожить до `MaxStep` — единственный успешный исход: любое падение шара
    /// закрывает эпизод как неуспех прямо в `OnActionReceived`.
    /// </summary>
    protected override bool TimeoutIsSuccess => true;

    Vector2 m_HeuristicTilt;

    public override void Initialize()
    {
        base.Initialize();
        if (area == null) area = GetComponentInParent<BallBalanceArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetEpisode();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Компоненты кватерниона, а не углы Эйлера: углы Эйлера разрывны
        // (359° и 1° — соседние наклоны, но далёкие числа), и сеть на таком
        // входе учится заметно хуже.
        var rotation = transform.localRotation;
        sensor.AddObservation(rotation.z);          // 1
        sensor.AddObservation(rotation.x);          // 1
        sensor.AddObservation(area.BallOffset);     // 3
        sensor.AddObservation(area.ball.linearVelocity);  // 3
        // Итого 8 — совпадает со строкой-контрактом ENV_SPEC.md.
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float actionZ = rotationSpeed * Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float actionX = rotationSpeed * Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        // Наклон ограничивается по обеим осям: без ограничения политика может
        // «перевернуть» платформу и удерживать шар на её ребре.
        var rotation = transform.localRotation;
        if ((rotation.z < tiltLimit && actionZ > 0f) || (rotation.z > -tiltLimit && actionZ < 0f))
            transform.Rotate(Vector3.forward, actionZ);

        if ((rotation.x < tiltLimit && actionX > 0f) || (rotation.x > -tiltLimit && actionX < 0f))
            transform.Rotate(Vector3.right, actionX);

        if (area.HasFallen())
        {
            AddReward(dropReward);
            TrackStep();
            EndEpisodeWithResult("Dropped", success: false);
            return;
        }

        AddReward(stepReward);
        TrackStep();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuous = actionsOut.ContinuousActions;
        continuous[0] = m_HeuristicTilt.x;
        continuous[1] = m_HeuristicTilt.y;
    }

    void Update()
    {
        // Ввод читается в Update, а расходуется в Heuristic: FixedUpdate может
        // пропустить короткое нажатие, а Update — нет.
        float z = 0f;
        float x = 0f;
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.leftArrowKey.isPressed || kb.aKey.isPressed) z = 1f;
        else if (kb.rightArrowKey.isPressed || kb.dKey.isPressed) z = -1f;
        if (kb.upArrowKey.isPressed || kb.wKey.isPressed) x = 1f;
        else if (kb.downArrowKey.isPressed || kb.sKey.isPressed) x = -1f;
#else
        if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) z = 1f;
        else if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) z = -1f;
        if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) x = 1f;
        else if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) x = -1f;
#endif
        m_HeuristicTilt = new Vector2(z, x);
    }
}
