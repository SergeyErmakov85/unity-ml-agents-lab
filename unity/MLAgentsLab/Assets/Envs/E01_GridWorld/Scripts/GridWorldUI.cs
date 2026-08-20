using TMPro;
using UnityEngine;

/// <summary>
/// Панель статистики (TS-001, раздел 11): каждый кадр читает состояние
/// агента и выводит episode / step / state / action / reward / result.
/// </summary>
public class GridWorldUI : MonoBehaviour
{
    public GridWorldAgent agent;

    public TextMeshProUGUI textEpisode;
    public TextMeshProUGUI textStep;
    public TextMeshProUGUI textState;
    public TextMeshProUGUI textLastAction;
    public TextMeshProUGUI textReward;
    public TextMeshProUGUI textResult;

    private void Update()
    {
        if (agent == null) return;

        textEpisode.text = $"Episode: {agent.CompletedEpisodes + 1}";
        textStep.text = $"Step: {agent.StepCount} / {agent.MaxStep}";
        textState.text = $"State: {agent.CurrentStateIndex}";
        textLastAction.text = $"Action: {agent.LastActionLabel}";
        textReward.text = $"Reward: {agent.GetCumulativeReward():F2}";
        textResult.text = $"Result: {agent.LastEpisodeResult}";
    }
}
