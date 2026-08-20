using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E00_Bandit` по TS-000: стол с K рычагами, агент над
/// ним, N тренировочных арен из одного префаба.
/// Меню: Tools → RL → Build Bandit Scene.
///
/// Сцена строится кодом целиком: чтобы её изменить, меняется этот скрипт
/// и сцена пересобирается. Так изменения читаемы в diff, а сцена
/// воспроизводима из чистого клона.
/// </summary>
public static class BanditSetup
{
    /// <summary>Идентификатор среды. Behavior Name в Unity обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E00_Bandit";

    private const string Root = "Assets/Envs/E00_Bandit";
    private const string ScenePath = Root + "/Scenes/E00_Bandit.unity";

    /// <summary>Число арен = число параллельных сред для Python (требование 8.2).</summary>
    private const int AreaCount = 8;

    /// <summary>Шаг размещения арен по X. Арена занимает 6 единиц, шаг 10 даёт зазор 4.</summary>
    private const float AreaSpacing = 10f;

    /// <summary>
    /// Эпизод бандита — ровно одно нажатие. MaxStep = 1 задан явно (требование 7.2),
    /// но до обрыва по нему дело не доходит: агент завершает эпизод сам, внутри
    /// `OnActionReceived`, а проверка MaxStep выполняется раньше — в фазе
    /// `AgentPreStep` следующего шага, когда счётчик уже сброшен.
    /// </summary>
    private const int MaxStep = 1;

    private const int DecisionPeriod = 1;

    /// <summary>Вероятности рук. Порядок и значения зафиксированы в TS-000, §5.</summary>
    private static readonly float[] ArmProbabilities = { 0.20f, 0.35f, 0.50f, 0.65f, 0.80f };

    private const float LeverSpacing = 1.2f;

    private static readonly Color ColTable = new Color(0.22f, 0.22f, 0.28f);
    private static readonly Color ColAgent = new Color(0.20f, 0.50f, 0.90f);

    [MenuItem("Tools/RL/Build Bandit Scene")]
    public static void BuildBanditScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<BanditArea>();
        areaScript.armProbabilities = (float[])ArmProbabilities.Clone();

        var table = GameObject.CreatePrimitive(PrimitiveType.Cube);
        table.name = "Table";
        table.transform.SetParent(area.transform, false);
        table.transform.localPosition = new Vector3(0f, -0.25f, 0f);
        table.transform.localScale = new Vector3(ArmProbabilities.Length * LeverSpacing + 1f, 0.5f, 2f);
        table.GetComponent<Renderer>().sharedMaterial = MakeMaterial("TableMat", ColTable);
        // Коллайдер не нужен: в среде нет физики (таксономия 13.3 — «UI-сцена без физики»).
        Object.DestroyImmediate(table.GetComponent<BoxCollider>());

        var levers = new GameObject("Arms");
        levers.transform.SetParent(area.transform, false);

        var visuals = new Transform[ArmProbabilities.Length];
        float offset = (ArmProbabilities.Length - 1) * 0.5f * LeverSpacing;
        for (int i = 0; i < ArmProbabilities.Length; i++)
        {
            var lever = GameObject.CreatePrimitive(PrimitiveType.Cube);
            lever.name = $"Arm_{i}";
            lever.transform.SetParent(levers.transform, false);
            lever.transform.localPosition = new Vector3(i * LeverSpacing - offset, 0.05f, 0f);
            lever.transform.localScale = new Vector3(0.8f, 0.1f, 0.8f);
            lever.GetComponent<Renderer>().sharedMaterial = MakeMaterial($"ArmMat_{i}", areaScript.ArmColor(i));
            Object.DestroyImmediate(lever.GetComponent<BoxCollider>());
            visuals[i] = lever.transform;
        }
        areaScript.armVisuals = visuals;

        var agentGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        agentGo.name = "Agent";
        agentGo.tag = "agent";
        agentGo.transform.SetParent(area.transform, false);
        agentGo.transform.localPosition = new Vector3(0f, 4f, -1.5f);
        agentGo.transform.localScale = Vector3.one * 0.8f;
        agentGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("AgentMat", ColAgent);
        Object.DestroyImmediate(agentGo.GetComponent<SphereCollider>());

        var agent = agentGo.AddComponent<BanditAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;

        var behavior = agentGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 1;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(ArmProbabilities.Length);
        behavior.BehaviorType = BehaviorType.Default;

        var requester = agentGo.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = DecisionPeriod;
        requester.TakeActionsBetweenDecisions = false;

        // ---------- Префаб и K арен ----------
        var areaPrefab = PrefabUtility.SaveAsPrefabAsset(area, Root + "/Prefabs/TrainingArea.prefab");
        Object.DestroyImmediate(area);

        var rootTrainingAreas = new GameObject("TrainingAreas");
        var rootCameras = new GameObject("Cameras");
        var rootLighting = new GameObject("Lighting");
        new GameObject("Environment");
        new GameObject("Managers");
        new GameObject("UI");
        new GameObject("Debug");

        for (int i = 0; i < AreaCount; i++)
        {
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(areaPrefab, rootTrainingAreas.transform);
            instance.name = $"TrainingArea_{i:00}";
            instance.transform.localPosition = new Vector3(i * AreaSpacing, 0f, 0f);
            instance.GetComponent<BanditArea>().areaIndex = i;  // смещение сида арены
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3((AreaCount - 1) * AreaSpacing * 0.5f, 8f, -14f);
        cam.transform.rotation = Quaternion.Euler(25f, 0f, 0f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.13f, 0.13f, 0.16f);

        // ---------- Освещение ----------
        var lightGo = GameObject.Find("Directional Light");
        lightGo.name = "DirectionalLight";
        lightGo.transform.SetParent(rootLighting.transform, true);
        var light = lightGo.GetComponent<Light>();
        light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        light.intensity = 1f;
        light.shadows = LightShadows.Soft;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.25f, 0.25f, 0.31f);

        EditorSceneManager.SaveScene(scene, ScenePath);
        AssetDatabase.SaveAssets();
        Debug.Log($"Bandit-сцена собрана: {ScenePath}, арен: {AreaCount}, рук: {ArmProbabilities.Length}");
    }

    private static Material MakeMaterial(string name, Color color)
    {
        string path = $"{Root}/Materials/{name}.mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            mat = new Material(shader) { color = color };
            AssetDatabase.CreateAsset(mat, path);
        }
        else
        {
            mat.color = color;
            EditorUtility.SetDirty(mat);
        }
        return mat;
    }

    /// <summary>Рекурсивно создаёт папку ассетов вида "Assets/a/b/c".</summary>
    private static void EnsureFolder(string path)
    {
        var parts = path.Split('/');
        var current = parts[0];
        for (int i = 1; i < parts.Length; i++)
        {
            var next = $"{current}/{parts[i]}";
            if (!AssetDatabase.IsValidFolder(next))
                AssetDatabase.CreateFolder(current, parts[i]);
            current = next;
        }
    }
}
