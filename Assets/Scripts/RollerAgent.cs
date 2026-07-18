using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Агент-шар, который учится докатываться до цели (Target), не падая с платформы.
///
/// Наблюдения (Space Size = 8):
///   - позиция цели (x, y, z)          — 3
///   - позиция агента (x, y, z)        — 3
///   - скорость агента (x, z)          — 2
///
/// Действия: Continuous 2 — сила по осям X и Z.
/// (Вариант Discrete 4 — движение в 4 стороны — см. закомментированный код внизу.)
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class RollerAgent : Agent
{
    [Header("Ссылки")]
    [Tooltip("Цель, до которой нужно докатиться")]
    public Transform target;

    [Header("Параметры движения")]
    public float forceMultiplier = 10f;

    [Header("Границы платформы (для респауна цели)")]
    public float spawnRange = 4f;

    private Rigidbody rBody;

    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Если агент упал с платформы — вернуть его на место и обнулить импульс
        if (transform.localPosition.y < 0f)
        {
            rBody.angularVelocity = Vector3.zero;
            rBody.linearVelocity = Vector3.zero;
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
        }

        // Переместить цель в случайную точку на платформе
        target.localPosition = new Vector3(
            Random.Range(-spawnRange, spawnRange),
            0.5f,
            Random.Range(-spawnRange, spawnRange));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(target.localPosition);     // 3
        sensor.AddObservation(transform.localPosition);  // 3
        sensor.AddObservation(rBody.linearVelocity.x);   // 1
        sensor.AddObservation(rBody.linearVelocity.z);   // 1
        // Итого: 8
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Continuous 2: сила по X и Z
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actions.ContinuousActions[0];
        controlSignal.z = actions.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        // Достигли цели — награда и новый эпизод
        float distanceToTarget = Vector3.Distance(transform.localPosition, target.localPosition);
        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        // Упали с платформы — эпизод окончен без награды
        else if (transform.localPosition.y < 0f)
        {
            EndEpisode();
        }
    }

    // Ручное управление (Behavior Type = Heuristic Only) — для проверки сцены стрелками/WASD
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxis("Horizontal");
        continuousActions[1] = Input.GetAxis("Vertical");
    }

    /*
    // ===== Вариант с Discrete-действиями (Branch Size = 4: вперёд/назад/влево/вправо) =====
    // В Behavior Parameters установите Discrete Branches = 1, Branch 0 Size = 4,
    // и замените OnActionReceived/Heuristic на:
    //
    // public override void OnActionReceived(ActionBuffers actions)
    // {
    //     Vector3 dir = actions.DiscreteActions[0] switch
    //     {
    //         0 => Vector3.forward,
    //         1 => Vector3.back,
    //         2 => Vector3.left,
    //         3 => Vector3.right,
    //         _ => Vector3.zero
    //     };
    //     rBody.AddForce(dir * forceMultiplier);
    //     ... (проверка цели/падения — та же)
    // }
    */
}
