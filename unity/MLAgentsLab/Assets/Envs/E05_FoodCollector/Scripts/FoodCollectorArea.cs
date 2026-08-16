using System.Collections.Generic;
using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E05_FoodCollector` (TS-005) — площадка 20×20 с рассыпанной
/// едой двух сортов.
///
/// Наследует <see cref="TrainingAreaBase"/>: сид приходит из Python, генератор
/// случайных чисел — свой у каждой арены.
///
/// **Съеденная еда не исчезает, а переставляется** в новую случайную точку.
/// Иначе плотность еды падает по ходу эпизода, задача незаметно меняется
/// на середине, и сравнивать первую половину эпизода со второй становится
/// нельзя.
/// </summary>
public class FoodCollectorArea : TrainingAreaBase
{
    [Header("Ссылки")]
    public Transform agent;
    public Transform foodRoot;

    [Header("Геометрия")]
    [Tooltip("Половина стороны площадки. Площадка 20×20 -> 10.")]
    public float halfSize = 10f;

    [Tooltip("Отступ от стен при выборе точки.")]
    public float margin = 1.5f;

    [Tooltip("Высота, на которой лежит еда и стоит агент.")]
    public float spawnHeight = 0.5f;

    [Tooltip("Минимальная дистанция между объектами при расстановке.")]
    public float minSeparation = 2f;

    readonly List<Vector3> m_Occupied = new List<Vector3>(32);

    /// <summary>Ставит агента и всю еду в случайные точки площадки.</summary>
    public void ResetEpisode(Rigidbody body)
    {
        body.angularVelocity = Vector3.zero;
        body.linearVelocity = Vector3.zero;

        m_Occupied.Clear();

        agent.localPosition = RandomFreePoint();
        agent.localRotation = Quaternion.Euler(0f, NextFloat(0f, 360f), 0f);
        m_Occupied.Add(agent.localPosition);

        for (int i = 0; i < foodRoot.childCount; i++)
        {
            var food = foodRoot.GetChild(i);
            food.gameObject.SetActive(true);
            food.localPosition = RandomFreePoint();
            m_Occupied.Add(food.localPosition);
        }
    }

    /// <summary>
    /// Переставляет съеденный кусок еды в новую свободную точку.
    ///
    /// Список занятых точек намеренно не пересобирается: он копится с начала
    /// эпизода и служит грубым «не ставить в кучу». Точность здесь не нужна,
    /// а пересборка на каждый съеденный кусок — работа в горячем пути.
    /// </summary>
    public void RespawnFood(Transform food)
    {
        food.localPosition = RandomFreePoint();
        m_Occupied.Add(food.localPosition);
    }

    Vector3 RandomFreePoint()
    {
        float extent = halfSize - margin;
        return SpawnService.RandomPointAwayFrom(
            Rng, Vector3.zero, new Vector2(extent, extent), spawnHeight,
            m_Occupied, minSeparation, out _);
    }
}
