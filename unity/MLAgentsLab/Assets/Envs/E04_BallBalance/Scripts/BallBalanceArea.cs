using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E04_BallBalance` (TS-004) — платформа с шаром.
///
/// Наследует <see cref="TrainingAreaBase"/>: сид и сложность приходят из Python
/// через `EnvironmentParametersChannel`, генератор случайных чисел — свой
/// у каждой арены (иначе при 8 аренах порядок обращений к глобальному
/// `UnityEngine.Random` зависел бы от порядка вызова `OnEpisodeBegin`).
///
/// Арена отвечает за расстановку и за проверку «шар упал». Наклон платформы
/// и награда — в агенте: наклон и есть действие.
/// </summary>
public class BallBalanceArea : TrainingAreaBase
{
    [Header("Ссылки")]
    public Transform platform;
    public Rigidbody ball;

    [Header("Старт эпизода")]
    [Tooltip("Половина стороны области, в которой появляется шар над платформой.")]
    public float spawnRange = 1.5f;

    [Tooltip("Высота появления шара над центром платформы.")]
    public float spawnHeight = 4f;

    [Tooltip("Максимальный случайный наклон платформы в начале эпизода, градусов.")]
    public float startTiltDegrees = 10f;

    [Header("Условие падения")]
    [Tooltip("Насколько ниже центра платформы должен опуститься шар, чтобы считаться упавшим.")]
    public float fallDepth = 1.5f;

    [Tooltip("Отклонение шара по X или Z от центра платформы, после которого он считается упавшим.")]
    public float fallRadius = 3f;

    /// <summary>Позиция шара относительно центра платформы — она же часть наблюдения.</summary>
    public Vector3 BallOffset => ball.position - platform.position;

    /// <summary>
    /// Ставит платформу горизонтально со случайным небольшим наклоном и роняет
    /// шар в случайную точку над ней.
    ///
    /// Случайный наклон нужен, чтобы агент не выучил единственную траекторию:
    /// без него каждый эпизод начинался бы из одного состояния, и политика
    /// оказалась бы хрупкой к любому отклонению.
    /// </summary>
    public void ResetEpisode()
    {
        platform.localRotation = Quaternion.Euler(
            NextFloat(-startTiltDegrees, startTiltDegrees),
            0f,
            NextFloat(-startTiltDegrees, startTiltDegrees));

        ball.angularVelocity = Vector3.zero;
        ball.linearVelocity = Vector3.zero;
        ball.position = platform.position + new Vector3(
            NextFloat(-spawnRange, spawnRange),
            spawnHeight,
            NextFloat(-spawnRange, spawnRange));
    }

    /// <summary>Упал ли шар с платформы.</summary>
    public bool HasFallen()
    {
        Vector3 offset = BallOffset;
        return offset.y < -fallDepth ||
               Mathf.Abs(offset.x) > fallRadius ||
               Mathf.Abs(offset.z) > fallRadius;
    }
}
