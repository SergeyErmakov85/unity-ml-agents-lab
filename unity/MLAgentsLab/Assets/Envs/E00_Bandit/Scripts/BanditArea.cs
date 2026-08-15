using LabRL.Core;
using UnityEngine;

/// <summary>
/// Арена среды `E00_Bandit` (TS-000) — один многорукий бандит.
///
/// Наследует <see cref="TrainingAreaBase"/>: сид приходит из Python через
/// `EnvironmentParametersChannel`, а генератор случайных чисел — свой у каждой
/// арены. Для бандита это критично вдвойне: вся среда состоит из случайности,
/// и общий на процесс `UnityEngine.Random` сделал бы прогон невоспроизводимым.
///
/// **Вероятности рук одинаковы во всех аренах и не зависят от сида.**
/// Иначе арены решали бы разные задачи: политика одна на все K арен, и «рука 4»
/// обязана означать одно и то же везде. Случайность арены — только в исходе
/// нажатия, а не в том, какая рука лучшая.
/// </summary>
public class BanditArea : TrainingAreaBase
{
    [Header("Руки")]
    [Tooltip("Вероятность награды 1.0 для каждой руки. Порядок зафиксирован в TS-000, §5.")]
    public float[] armProbabilities = { 0.20f, 0.35f, 0.50f, 0.65f, 0.80f };

    [Tooltip("Визуальные рычаги: по одному на руку, в том же порядке.")]
    public Transform[] armVisuals;

    [Header("Визуализация")]
    [Tooltip("Высота рычага при доле нажатий 1.0. Высота = доля нажатий этой руки.")]
    public float maxLeverHeight = 3f;

    [Tooltip("Цвет рычага в момент нажатия; гаснет за столько шагов.")]
    public float highlightDecay = 0.15f;

    /// <summary>Число рук.</summary>
    public int ArmCount => armProbabilities.Length;

    /// <summary>Индекс руки с наибольшей вероятностью — эталон для метрики «доля оптимальных выборов».</summary>
    public int BestArm { get; private set; }

    /// <summary>Вероятность лучшей руки: верхняя граница средней награды за шаг.</summary>
    public float BestProbability => armProbabilities[BestArm];

    int[] m_PullCounts;
    int m_TotalPulls;
    MaterialPropertyBlock m_Block;
    float[] m_Highlight;
    Renderer[] m_ArmRenderers;

    protected override void OnAreaInitialized()
    {
        m_PullCounts = new int[ArmCount];
        m_Highlight = new float[ArmCount];
        m_TotalPulls = 0;

        // Рендереры кэшируются один раз: визуализация обновляется каждый кадр
        // в каждой из восьми арен, и поиск компонента в этом цикле — лишняя
        // работа в горячем пути (требование 7.5).
        if (armVisuals != null)
        {
            m_ArmRenderers = new Renderer[armVisuals.Length];
            for (int i = 0; i < armVisuals.Length; i++)
            {
                if (armVisuals[i] != null) armVisuals[i].TryGetComponent(out m_ArmRenderers[i]);
            }
        }

        BestArm = 0;
        for (int i = 1; i < ArmCount; i++)
        {
            if (armProbabilities[i] > armProbabilities[BestArm]) BestArm = i;
        }
    }

    /// <summary>
    /// Нажимает руку и возвращает награду: 1.0 с вероятностью `armProbabilities[arm]`,
    /// иначе 0.0.
    ///
    /// Награда бернуллиевская, а не гауссова, сознательно: сопряжённое
    /// априорное распределение для неё — Beta, и Thompson Sampling выводится
    /// в две строки, без приближений (см. `labrl.algos.bandits`).
    /// </summary>
    public float Pull(int arm)
    {
        if (arm < 0 || arm >= ArmCount)
        {
            Debug.LogError($"{name}: рука {arm} вне диапазона [0, {ArmCount})", this);
            return 0f;
        }

        m_PullCounts[arm]++;
        m_TotalPulls++;
        m_Highlight[arm] = 1f;

        return NextFloat() < armProbabilities[arm] ? 1f : 0f;
    }

    /// <summary>Доля нажатий, пришедшихся на руку — для визуализации и отладки.</summary>
    public float PullShare(int arm) => m_TotalPulls == 0 ? 0f : (float)m_PullCounts[arm] / m_TotalPulls;

    void LateUpdate()
    {
        // Визуализация обновляется в LateUpdate, а не в Pull(): за один кадр
        // при time_scale=20 происходит много нажатий, и трогать трансформы
        // на каждом из них — лишняя работа в горячем пути (требование 7.5).
        if (armVisuals == null || m_PullCounts == null) return;

        m_Block ??= new MaterialPropertyBlock();

        for (int i = 0; i < armVisuals.Length && i < ArmCount; i++)
        {
            var lever = armVisuals[i];
            if (lever == null) continue;

            float height = Mathf.Max(0.1f, PullShare(i) * maxLeverHeight);
            var scale = lever.localScale;
            lever.localScale = new Vector3(scale.x, height, scale.z);
            lever.localPosition = new Vector3(lever.localPosition.x, height * 0.5f, lever.localPosition.z);

            m_Highlight[i] = Mathf.Max(0f, m_Highlight[i] - highlightDecay);
            var armRenderer = m_ArmRenderers != null && i < m_ArmRenderers.Length ? m_ArmRenderers[i] : null;
            if (armRenderer != null)
            {
                armRenderer.GetPropertyBlock(m_Block);
                m_Block.SetColor("_BaseColor", Color.Lerp(ArmColor(i), Color.white, m_Highlight[i]));
                armRenderer.SetPropertyBlock(m_Block);
            }
        }
    }

    /// <summary>Цвет руки: от красного (плохая) к зелёному (хорошая). Только для наглядности.</summary>
    public Color ArmColor(int arm)
    {
        float t = ArmCount < 2 ? 1f : (float)arm / (ArmCount - 1);
        return Color.Lerp(new Color(0.75f, 0.25f, 0.25f), new Color(0.25f, 0.75f, 0.35f), t);
    }
}
