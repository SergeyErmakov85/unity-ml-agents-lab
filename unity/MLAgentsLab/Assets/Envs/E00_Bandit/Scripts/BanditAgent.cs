using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент многорукого бандита (TS-000).
///
/// Бандит — это MDP с **одним** состоянием и эпизодом длины 1: агент выбирает
/// руку, получает награду и эпизод заканчивается. Никакой динамики нет,
/// поэтому нет и задачи «предсказать будущее»: остаётся ровно один вопрос —
/// как делить бюджет попыток между проверкой неизвестного (exploration)
/// и использованием лучшего известного (exploitation). Урок 1.7.
///
/// Наблюдение — одно число, **константа 1.0**. Это не заглушка: состояние
/// действительно одно, а единица делает наблюдение корректным one-hot вектором
/// длины 1. Благодаря этому таблица ценностей рук `Q` формы (1, K) уезжает
/// в ONNX тем же линейным слоем, что и в `E01_GridWorld`:
/// `onehot(s) @ Qᵀ == Q[s]` (см. `labrl.nets.tabular.OneHotQTable`).
/// Ноль на этом месте обнулил бы выход слоя — отсюда именно 1.0.
/// </summary>
public class BanditAgent : AgentBase
{
    [Header("Ссылки")]
    public BanditArea area;

    public override string EnvId => "E00_Bandit";

    int m_HeuristicArm;

    public override void Initialize()
    {
        base.Initialize();
        if (area == null) area = GetComponentInParent<BanditArea>();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Единственное состояние среды. obs_size = 1 — см. строку-контракт ENV_SPEC.md.
        sensor.AddObservation(1f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int arm = actions.DiscreteActions[0];

        float reward = area.Pull(arm);
        AddReward(reward);
        TrackStep();

        bool optimal = arm == area.BestArm;
        // Сожаление (regret) шага — разница между лучшей достижимой средней
        // наградой и средней наградой выбранной руки. Именно его минимизируют
        // алгоритмы бандитов, и именно оно, а не сырая награда, показывает
        // качество разведки: награда шумная, сожаление — нет.
        MetricsRecorder.Average("Regret", area.BestProbability - area.armProbabilities[arm]);

        // Доля оптимальных выборов уходит в Env/SuccessRate внутри
        // EndEpisodeWithResult — отдельной метрики для неё не требуется.
        EndEpisodeWithResult(optimal ? "Optimal" : "Suboptimal", optimal);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        discrete[0] = m_HeuristicArm;
    }

    void Update()
    {
        // Ввод читается в Update: короткое нажатие клавиши FixedUpdate пропустит.
        // Клавиши 1..9 — номер руки (нумерация для человека с единицы).
        int count = Mathf.Min(area.ArmCount, 9);
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        for (int i = 0; i < count; i++)
        {
            if (kb[Key.Digit1 + i].isPressed) m_HeuristicArm = i;
        }
#else
        for (int i = 0; i < count; i++)
        {
            if (Input.GetKey(KeyCode.Alpha1 + i)) m_HeuristicArm = i;
        }
#endif
    }
}
