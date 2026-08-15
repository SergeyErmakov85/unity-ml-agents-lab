using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E02_CartPoleUnity` (TS-002) — тележка с шестом.
///
/// Динамика **не** отдана физическому движку: уравнения интегрируются вручную,
/// ровно теми же формулами и константами, что в классической реализации
/// `CartPole-v0` (Barto, Sutton & Anderson, 1983; Gymnasium `cartpole.py`).
/// Причина принципиальная: урок 1.5 переносит в Unity именно эталонную задачу,
/// и агент, обученный здесь, обязан вести себя так же, как в Gymnasium.
/// PhysX дал бы другую задачу — с трением, демпфированием и недетерминизмом
/// между запусками, из-за которого прогон с фиксированным сидом не повторяется.
///
/// Визуальные объекты (`Cart`, `PolePivot`) — отражение состояния, а не его
/// источник: их трансформы переписываются из (x, θ) после каждого шага.
/// </summary>
public class CartPoleArea : TrainingAreaBase
{
    [Header("Ссылки")]
    public Transform cart;
    public Transform polePivot;

    [Header("Физические константы (Gymnasium CartPole-v0)")]
    [Tooltip("Ускорение свободного падения, м/с².")]
    public float gravity = 9.8f;

    [Tooltip("Масса тележки, кг.")]
    public float massCart = 1.0f;

    [Tooltip("Масса шеста, кг.")]
    public float massPole = 0.1f;

    [Tooltip("ПОЛОВИНА длины шеста, м. Полная длина шеста — 2·length.")]
    public float length = 0.5f;

    [Tooltip("Модуль силы, прикладываемой к тележке, Н.")]
    public float forceMagnitude = 10.0f;

    [Tooltip("Шаг интегрирования, с. Совпадает с шагом Academy при DecisionPeriod = 1.")]
    public float tau = 0.02f;

    [Header("Границы эпизода")]
    [Tooltip("Максимальное отклонение тележки от центра, м.")]
    public float xThreshold = 2.4f;

    [Tooltip("Максимальный наклон шеста, градусов.")]
    public float thetaThresholdDegrees = 12f;

    [Header("Старт эпизода")]
    [Tooltip("Все четыре переменные состояния стартуют из U(-range, +range).")]
    public float startRange = 0.05f;

    /// <summary>Положение тележки, м.</summary>
    public float X { get; private set; }

    /// <summary>Скорость тележки, м/с.</summary>
    public float XDot { get; private set; }

    /// <summary>Угол отклонения шеста от вертикали, рад. Положительный — вправо (+X).</summary>
    public float Theta { get; private set; }

    /// <summary>Угловая скорость шеста, рад/с.</summary>
    public float ThetaDot { get; private set; }

    float TotalMass => massCart + massPole;
    float PoleMassLength => massPole * length;

    /// <summary>Порог наклона в радианах — сравнивать удобнее в них, задавать в градусах.</summary>
    public float ThetaThreshold => thetaThresholdDegrees * Mathf.Deg2Rad;

    /// <summary>Вышло ли состояние за границы: тележка уехала или шест упал.</summary>
    public bool IsFailed => Mathf.Abs(X) > xThreshold || Mathf.Abs(Theta) > ThetaThreshold;

    /// <summary>
    /// Ставит систему в стартовое состояние: все четыре переменные из U(−0.05, 0.05).
    /// Генератор — свой у арены (см. <see cref="TrainingAreaBase"/>), поэтому
    /// восемь арен стартуют по-разному, но воспроизводимо.
    /// </summary>
    public void ResetState()
    {
        X = NextFloat(-startRange, startRange);
        XDot = NextFloat(-startRange, startRange);
        Theta = NextFloat(-startRange, startRange);
        ThetaDot = NextFloat(-startRange, startRange);
        ApplyToVisuals();
    }

    /// <summary>
    /// Один шаг динамики. Действие 0 — сила влево, 1 — вправо.
    ///
    /// Уравнения (полудлина шеста l, масса шеста m, масса тележки M, сила F):
    /// <code>
    /// temp   = (F + m·l·θ̇²·sin θ) / (M + m)
    /// θ̈      = (g·sin θ − cos θ·temp) / (l·(4/3 − m·cos²θ/(M + m)))
    /// ẍ      = temp − m·l·θ̈·cos θ / (M + m)
    /// </code>
    /// Интегрирование — явное по Эйлеру, как в эталоне: сначала обновляются
    /// координаты по старым скоростям, затем скорости по ускорениям.
    /// </summary>
    public void Step(int action)
    {
        float force = action == 1 ? forceMagnitude : -forceMagnitude;
        float cosTheta = Mathf.Cos(Theta);
        float sinTheta = Mathf.Sin(Theta);

        float temp = (force + PoleMassLength * ThetaDot * ThetaDot * sinTheta) / TotalMass;
        float thetaAcc = (gravity * sinTheta - cosTheta * temp) /
                         (length * (4.0f / 3.0f - massPole * cosTheta * cosTheta / TotalMass));
        float xAcc = temp - PoleMassLength * thetaAcc * cosTheta / TotalMass;

        X += tau * XDot;
        XDot += tau * xAcc;
        Theta += tau * ThetaDot;
        ThetaDot += tau * thetaAcc;

        ApplyToVisuals();
    }

    /// <summary>Переносит состояние на трансформы. Визуализация — следствие, а не источник.</summary>
    void ApplyToVisuals()
    {
        if (cart != null)
        {
            var p = cart.localPosition;
            cart.localPosition = new Vector3(X, p.y, p.z);
        }

        if (polePivot != null)
        {
            // Пивот шеста — не потомок тележки, а её сосед: вращение внутри
            // неравномерно отмасштабированного родителя даёт сдвиговую
            // деформацию, и шест выглядел бы скошенным. Поэтому позиция
            // переносится вручную, а масштаб пивота остаётся единичным.
            var p = polePivot.localPosition;
            polePivot.localPosition = new Vector3(X, p.y, p.z);

            // Поворот вокруг +Z переводит +Y в −X, поэтому положительному θ
            // («шест наклонён вправо, к +X») соответствует угол −θ.
            polePivot.localRotation = Quaternion.Euler(0f, 0f, -Theta * Mathf.Rad2Deg);
        }
    }
}
