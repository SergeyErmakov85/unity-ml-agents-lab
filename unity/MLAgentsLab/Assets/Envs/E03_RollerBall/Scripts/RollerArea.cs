using LabRL.Core;
using UnityEngine;

/// <summary>
/// Тренировочная арена среды `E03_RollerBall` (TS-003).
///
/// Наследует <see cref="TrainingAreaBase"/>: сид и сложность приходят из Python
/// через `EnvironmentParametersChannel`, а генератор случайных чисел — свой
/// у каждой арены. Это принципиально: при 8 аренах глобальный
/// `UnityEngine.Random` дал бы порядок обращений, зависящий от порядка вызова
/// `OnEpisodeBegin`, и прогон перестал бы воспроизводиться при фиксированном сиде.
///
/// Арена отвечает только за расстановку: где появляется цель и куда
/// возвращается агент. Логика награды и завершения — в агенте.
/// </summary>
public class RollerArea : TrainingAreaBase
{
    [Header("Ссылки")]
    public Transform agent;
    public Transform target;

    [Header("Спавн цели")]
    [Tooltip("Половина стороны области спавна цели по X и Z.")]
    public float spawnRange = 4f;

    [Tooltip("Минимальная дистанция от агента до цели: ближе эпизод решается одним движением.")]
    public float minTargetDistance = 2f;

    [Header("Старт агента")]
    public Vector3 agentStartLocalPosition = new Vector3(0f, 0.5f, 0f);

    /// <summary>Высота, ниже которой агент считается упавшим с платформы.</summary>
    public float fallHeight = 0f;

    /// <summary>
    /// Ставит цель в случайную точку арены, не ближе <see cref="minTargetDistance"/>
    /// к стартовой позиции агента.
    ///
    /// Ограничение по числу попыток намеренное: при слишком большом
    /// `minTargetDistance` среда должна деградировать предсказуемо, а не зависать.
    /// Если условие не выполнено, берётся последняя точка — эпизод будет проще,
    /// но прогон продолжится.
    /// </summary>
    public void PlaceTarget()
    {
        var halfExtents = new Vector2(spawnRange, spawnRange);
        var center = transform.TransformPoint(agentStartLocalPosition);

        var occupied = s_OccupiedBuffer;
        occupied[0] = center;

        Vector3 world = SpawnService.RandomPointAwayFrom(
            Rng, center, halfExtents, center.y, occupied, minTargetDistance, out _);

        target.position = world;
    }

    /// <summary>Возвращает агента в стартовую точку и гасит импульс.</summary>
    public void ResetAgent(Rigidbody body)
    {
        body.angularVelocity = Vector3.zero;
        body.linearVelocity = Vector3.zero;
        agent.localPosition = agentStartLocalPosition;
    }

    /// <summary>Упал ли агент с платформы.</summary>
    public bool HasFallen() => agent.localPosition.y < fallHeight;

    // Один буфер на процесс: метод вызывается каждый эпизод в каждой арене,
    // а аллокации в горячем пути запрещены (требование 7.5).
    static readonly Vector3[] s_OccupiedBuffer = new Vector3[1];
}
