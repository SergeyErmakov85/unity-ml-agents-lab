using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент задачи «тележка с шестом» (TS-002) — перенос эталонного `CartPole-v0`
/// в Unity, урок 1.5.
///
/// Учебный смысл среды: это первая среда с **непрерывным** наблюдением, где
/// табличный метод в лоб уже не работает. Ответ урока — дискретизация:
/// четырёхмерный вектор режется на ячейки, и каждая ячейка становится
/// состоянием таблицы. Отсюда прямой мост к DQN (`E03`), который заменяет
/// ручную сетку обучаемым признаковым представлением.
///
/// Наблюдение отдаётся **сырым**, без нормализации и без дискретизации:
/// сетка — свойство алгоритма, а не среды, и она обязана быть частью
/// экспортируемого графа ONNX (требование 10.7), иначе Unity и Python
/// разойдутся. См. `labrl.nets.discretized.DiscretizedQTable`.
/// </summary>
public class CartPoleAgent : AgentBase
{
    [Header("Ссылки")]
    public CartPoleArea area;

    [Header("Награды")]
    [Tooltip("Награда за каждый шаг, на котором система ещё не упала. Эталон CartPole: +1.")]
    public float stepReward = 1f;

    public override string EnvId => "E02_CartPoleUnity";

    /// <summary>
    /// Дожить до `MaxStep` — единственный успешный исход этой среды: падение
    /// шеста и выезд за границу закрываются как неуспех прямо в
    /// `OnActionReceived`, а всё остальное — обрыв по времени.
    /// </summary>
    protected override bool TimeoutIsSuccess => true;

    int m_HeuristicAction;

    public override void Initialize()
    {
        base.Initialize();
        if (area == null) area = GetComponentInParent<CartPoleArea>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        area.ResetState();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(area.X);         // положение тележки, м
        sensor.AddObservation(area.XDot);      // скорость тележки, м/с
        sensor.AddObservation(area.Theta);     // угол шеста, рад
        sensor.AddObservation(area.ThetaDot);  // угловая скорость шеста, рад/с
        // Итого 4 — совпадает со строкой-контрактом ENV_SPEC.md.
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int action = actions.DiscreteActions[0];
        area.Step(action);

        // Награда даётся за шаг, включая тот, на котором система упала, —
        // так же, как в эталонной реализации. Поэтому суммарная награда
        // эпизода численно равна числу прожитых шагов.
        AddReward(stepReward);
        TrackStep();

        if (area.IsFailed)
        {
            EndEpisodeWithResult(Mathf.Abs(area.Theta) > area.ThetaThreshold ? "PoleFell" : "OutOfBounds",
                                 success: false);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        discrete[0] = m_HeuristicAction;
    }

    void Update()
    {
        // Ввод читается в Update, а расходуется в Heuristic: FixedUpdate может
        // пропустить короткое нажатие, а Update — нет.
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.leftArrowKey.isPressed || kb.aKey.isPressed) m_HeuristicAction = 0;
        else if (kb.rightArrowKey.isPressed || kb.dKey.isPressed) m_HeuristicAction = 1;
#else
        if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) m_HeuristicAction = 0;
        else if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) m_HeuristicAction = 1;
#endif
    }
}
