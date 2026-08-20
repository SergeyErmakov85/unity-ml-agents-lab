using LabRL.Core;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Агент GridWorld (TS-001): дискретное состояние 0–24 (one-hot 25),
/// четыре действия (N/S/E/W), перемещение телепортом между центрами клеток.
///
/// Наследует <see cref="AgentBase"/>: оттуда — проверка совпадения Behavior Name
/// с идентификатором среды и публикация метрик эпизода в неймспейс Env/.
///
/// Свойство <see cref="CurrentStateIndex"/> открыто для внешнего табличного
/// Q-learning: он работает с индексом состояния, а не с one-hot вектором.
/// </summary>
public class GridWorldAgent : AgentBase
{
    public GridWorldEnvironment env;

    public override string EnvId => "E01_GridWorld";

    private Vector2Int currentCell;

    // Буфер ручного управления: стрелка, нажатая в Update, расходуется
    // одним решением в Heuristic (одна клетка за нажатие).
    private int bufferedAction = -1;
    private bool heuristicIdle;

    public int CurrentStateIndex => env.StateIndex(currentCell);
    public string LastActionLabel { get; private set; } = "-";

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        currentCell = env.randomStart ? env.RandomFreeCell() : env.startCell;
        transform.position = env.CellToWorld(currentCell, 0.3f);
        LastActionLabel = "-";
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        int state = CurrentStateIndex;
        int size = env.cols * env.rows;
        for (int i = 0; i < size; i++)
            sensor.AddObservation(i == state ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (heuristicIdle)
        {
            heuristicIdle = false;
            return;
        }

        int a = actions.DiscreteActions[0];
        LastActionLabel = a switch { 0 => "N", 1 => "S", 2 => "E", 3 => "W", _ => "-" };

        currentCell = env.ResolveMove(currentCell, a);
        transform.position = env.CellToWorld(currentCell, 0.3f);

        AddReward(env.stepReward);
        // Обрыв по MaxStep выполняет сам ML-Agents, раньше следующего вызова
        // OnActionReceived. Чтобы такой эпизод не потерялся в статистике,
        // база запоминает награду и шаги на каждом шаге (см. AgentBase.TrackStep).
        TrackStep();

        if (env.IsGoal(currentCell))
        {
            AddReward(env.goalReward);
            EndEpisodeWithResult("Goal", success: true);
        }
        else if (env.IsTrap(currentCell))
        {
            AddReward(env.trapReward);
            EndEpisodeWithResult("Trap", success: false);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discrete = actionsOut.DiscreteActions;
        if (bufferedAction >= 0)
        {
            discrete[0] = bufferedAction;
            bufferedAction = -1;
        }
        else
        {
            discrete[0] = 0;
            heuristicIdle = true; // решение без нажатия — ход пропускается
        }
    }

    private void Update()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.upArrowKey.wasPressedThisFrame) bufferedAction = 0;
        else if (kb.downArrowKey.wasPressedThisFrame) bufferedAction = 1;
        else if (kb.rightArrowKey.wasPressedThisFrame) bufferedAction = 2;
        else if (kb.leftArrowKey.wasPressedThisFrame) bufferedAction = 3;
#else
        if (Input.GetKeyDown(KeyCode.UpArrow)) bufferedAction = 0;
        else if (Input.GetKeyDown(KeyCode.DownArrow)) bufferedAction = 1;
        else if (Input.GetKeyDown(KeyCode.RightArrow)) bufferedAction = 2;
        else if (Input.GetKeyDown(KeyCode.LeftArrow)) bufferedAction = 3;
#endif
    }
}
