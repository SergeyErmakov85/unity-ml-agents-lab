using System.Collections.Generic;
using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E06_Hunter3D` (TS-006) — площадка 20×20 с препятствиями,
/// охотником и целью.
///
/// Наследует <see cref="TrainingAreaBase"/>: сид и сложность приходят из Python
/// через `EnvironmentParametersChannel`, генератор случайных чисел — свой
/// у каждой арены.
///
/// Разделение ответственности такое же, как в остальных средах: арена
/// отвечает за расстановку и геометрию, агент — за наблюдения, действия
/// и награду.
///
/// **Препятствия расставляются один раз при инициализации арены, а не каждый
/// эпизод.** Это осознанный выбор: меняющаяся от эпизода к эпизоду планировка —
/// это уже Domain Randomization, предмет `E09_CurriculumMaze`. Здесь агент
/// учится обходить фиксированную планировку, а меняются только стартовые
/// точки охотника и цели.
/// </summary>
public class HunterArea : TrainingAreaBase
{
    [Header("Ссылки")]
    public Transform hunter;
    public Transform target;
    public Transform obstacleRoot;

    [Header("Геометрия")]
    [Tooltip("Половина стороны площадки. Площадка 20×20 -> 10.")]
    public float halfSize = 10f;

    [Tooltip("Отступ от стен при выборе точки спавна.")]
    public float spawnMargin = 1.5f;

    [Tooltip("Высота, на которой стоят охотник и цель.")]
    public float spawnHeight = 0.5f;

    [Header("Спавн")]
    [Tooltip("Минимальная дистанция между охотником и целью при старте эпизода. " +
             "Без неё часть эпизодов решается одним движением и завышает оценку.")]
    public float minSpawnSeparation = 8f;

    [Tooltip("Минимальная дистанция от точки спавна до центра препятствия.")]
    public float obstacleClearance = 2.5f;

    /// <summary>
    /// Максимальное расстояние между двумя точками площадки — диагональ.
    /// Нормировочный множитель потенциала Φ(s) = −d(s)/maxDistance
    /// и наблюдений: без него величины зависели бы от размера арены.
    /// </summary>
    public float MaxDistance => 2f * halfSize * Mathf.Sqrt(2f);

    /// <summary>Текущее расстояние между охотником и целью, м.</summary>
    public float Distance => Vector3.Distance(hunter.localPosition, target.localPosition);

    readonly List<Vector3> m_Occupied = new List<Vector3>(8);
    Vector3[] m_ObstacleCenters = System.Array.Empty<Vector3>();

    protected override void OnAreaInitialized()
    {
        if (obstacleRoot == null) return;

        m_ObstacleCenters = new Vector3[obstacleRoot.childCount];
        for (int i = 0; i < obstacleRoot.childCount; i++)
            m_ObstacleCenters[i] = obstacleRoot.GetChild(i).localPosition;
    }

    /// <summary>
    /// Ставит охотника и цель в случайные точки площадки: врозь друг от друга
    /// и не внутри препятствий.
    /// </summary>
    public void ResetEpisode(Rigidbody body)
    {
        body.angularVelocity = Vector3.zero;
        body.linearVelocity = Vector3.zero;

        float extent = halfSize - spawnMargin;
        var halfExtents = new Vector2(extent, extent);

        m_Occupied.Clear();
        for (int i = 0; i < m_ObstacleCenters.Length; i++)
            m_Occupied.Add(m_ObstacleCenters[i]);

        Vector3 hunterPoint = SpawnService.RandomPointAwayFrom(
            Rng, Vector3.zero, halfExtents, spawnHeight, m_Occupied, obstacleClearance, out _);
        hunter.localPosition = hunterPoint;
        hunter.localRotation = Quaternion.Euler(0f, NextFloat(0f, 360f), 0f);

        // Цель отодвигается и от препятствий, и от охотника, поэтому минимальная
        // дистанция берётся наибольшая из двух требований.
        m_Occupied.Add(hunterPoint);
        target.localPosition = SpawnService.RandomPointAwayFrom(
            Rng, Vector3.zero, halfExtents, spawnHeight, m_Occupied,
            Mathf.Max(obstacleClearance, minSpawnSeparation), out _);
    }
}
