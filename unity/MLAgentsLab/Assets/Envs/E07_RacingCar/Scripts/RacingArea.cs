using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E07_RacingCar` (TS-007) — замкнутый кольцевой трек
/// с контрольными точками.
///
/// Наследует <see cref="TrainingAreaBase"/>: сид и сложность приходят из Python,
/// генератор случайных чисел — свой у каждой арены.
///
/// Зачем нужны контрольные точки. Без них «проехать круг» — событие, которое
/// случайная политика не наступит никогда, и обучение стоит на месте. Точки
/// разбивают круг на восемь достижимых подзадач и дают редкую, но регулярную
/// награду. Засчитываются они **строго по порядку**: иначе агент научился бы
/// крутиться возле одной точки, пересекая её туда-обратно.
/// </summary>
public class RacingArea : TrainingAreaBase
{
    [Header("Ссылки")]
    public Transform car;
    public Transform checkpointRoot;

    [Header("Геометрия трека")]
    [Tooltip("Радиус осевой линии трека, м.")]
    public float trackRadius = 11.5f;

    [Tooltip("Ширина полотна трека, м.")]
    public float trackWidth = 5f;

    [Tooltip("Разброс стартового положения вдоль радиуса, доля от полуширины.")]
    public float startRadialJitter = 0.5f;

    [Tooltip("Разброс стартового курса, градусов.")]
    public float startHeadingJitter = 15f;

    /// <summary>Сколько контрольных точек на круге.</summary>
    public int CheckpointCount => checkpointRoot == null ? 0 : checkpointRoot.childCount;

    /// <summary>Номер следующей ожидаемой контрольной точки.</summary>
    public int NextCheckpoint { get; private set; }

    /// <summary>Длина круга по осевой линии — нормировка расстояний в наблюдении.</summary>
    public float LapLength => 2f * Mathf.PI * trackRadius;

    /// <summary>Позиция следующей контрольной точки в системе координат арены.</summary>
    public Vector3 NextCheckpointPosition =>
        checkpointRoot.GetChild(Mathf.Min(NextCheckpoint, CheckpointCount - 1)).localPosition;

    /// <summary>
    /// Ставит машину на старт: на осевой линии в нулевом угле, с небольшим
    /// разбросом по радиусу и курсу.
    ///
    /// Разброс обязателен: без него каждый эпизод начинается из одного
    /// состояния, и политика оказывается хрупкой к любому отклонению —
    /// а отклонения неизбежны уже на первом повороте.
    /// </summary>
    public void ResetEpisode(Rigidbody body)
    {
        body.angularVelocity = Vector3.zero;
        body.linearVelocity = Vector3.zero;

        float radius = trackRadius + NextFloat(-1f, 1f) * startRadialJitter * trackWidth * 0.5f;
        car.localPosition = new Vector3(radius, 0.5f, 0f);
        // Машина смотрит вдоль трека — против часовой стрелки, то есть в +Z.
        car.localRotation = Quaternion.Euler(0f, NextFloat(-startHeadingJitter, startHeadingJitter), 0f);

        NextCheckpoint = 0;
    }

    /// <summary>
    /// Регистрирует пересечение контрольной точки.
    /// </summary>
    /// <returns>
    /// `true`, если точка была той, которую ждали, и счётчик продвинулся.
    /// Пересечение любой другой точки игнорируется — именно это не даёт
    /// набирать награду, катаясь взад-вперёд через одну и ту же точку.
    /// </returns>
    public bool TryPassCheckpoint(Transform checkpoint)
    {
        if (NextCheckpoint >= CheckpointCount) return false;
        if (checkpointRoot.GetChild(NextCheckpoint) != checkpoint) return false;

        NextCheckpoint++;
        return true;
    }

    /// <summary>Пройден ли полный круг.</summary>
    public bool LapCompleted => NextCheckpoint >= CheckpointCount;

    /// <summary>Доля пройденного круга ∈ [0, 1] — признак наблюдения.</summary>
    public float Progress => CheckpointCount == 0 ? 0f : (float)NextCheckpoint / CheckpointCount;
}
