using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E03_RollerBall` по TS-003: платформа со стенами,
/// цель, агент-шар, K тренировочных арен из одного префаба.
/// Меню: Tools → RL → Build Training Scene.
///
/// Сцена строится кодом целиком: чтобы её изменить, меняется этот скрипт
/// и сцена пересобирается. Так изменения читаемы в diff, а сцена
/// воспроизводима из чистого клона.
/// </summary>
public static class RLEnvironmentSetup
{
    /// <summary>Идентификатор среды. Behavior Name в Unity обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E03_RollerBall";

    private const string Root = "Assets/Envs/E03_RollerBall";
    private const string ScenePath = Root + "/Scenes/E03_RollerBall.unity";

    /// <summary>Число арен = число параллельных сред для Python (требование 8.2).</summary>
    private const int AreaCount = 8;

    /// <summary>
    /// Шаг размещения арен по X. Арена занимает 11 единиц (платформа со стенами),
    /// шаг 16 оставляет зазор 5: коллайдеры соседних арен не пересекаются.
    /// </summary>
    private const float AreaSpacing = 16f;

    private const int MaxStep = 1000;
    private const int DecisionPeriod = 5;

    private static readonly Color ColFloor = new Color(0.35f, 0.35f, 0.40f);
    private static readonly Color ColWall = new Color(0.60f, 0.60f, 0.65f);
    private static readonly Color ColTarget = new Color(0.20f, 0.80f, 0.30f);
    private static readonly Color ColAgent = new Color(0.20f, 0.50f, 0.90f);

    [MenuItem("Tools/RL/Build Training Scene")]
    public static void BuildTrainingScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<RollerArea>();

        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.SetParent(area.transform, false);
        floor.GetComponent<Renderer>().sharedMaterial = MakeMaterial("FloorMat", ColFloor);

        var walls = new GameObject("Walls");
        walls.transform.SetParent(area.transform, false);
        CreateWall(walls.transform, "Wall_North", new Vector3(0f, 0.5f, 5.25f), new Vector3(11f, 1f, 0.5f));
        CreateWall(walls.transform, "Wall_South", new Vector3(0f, 0.5f, -5.25f), new Vector3(11f, 1f, 0.5f));
        CreateWall(walls.transform, "Wall_East", new Vector3(5.25f, 0.5f, 0f), new Vector3(0.5f, 1f, 11f));
        CreateWall(walls.transform, "Wall_West", new Vector3(-5.25f, 0.5f, 0f), new Vector3(0.5f, 1f, 11f));

        var target = GameObject.CreatePrimitive(PrimitiveType.Cube);
        target.name = "Target";
        target.tag = "goal";
        target.transform.SetParent(area.transform, false);
        target.transform.localPosition = new Vector3(3f, 0.5f, 3f);
        target.GetComponent<Renderer>().sharedMaterial = MakeMaterial("TargetMat", ColTarget);
        // Цель — только визуальный и геометрический ориентир: столкновение
        // с ней не обрабатывается физикой, дистанция считается в коде агента.
        target.GetComponent<BoxCollider>().isTrigger = true;

        var agentGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        agentGo.name = "Agent";
        agentGo.tag = "agent";
        agentGo.transform.SetParent(area.transform, false);
        agentGo.transform.localPosition = new Vector3(0f, 0.5f, 0f);
        agentGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("AgentMat", ColAgent);

        var body = agentGo.AddComponent<Rigidbody>();
        body.mass = 1f;

        var agent = agentGo.AddComponent<RollerAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;
        agent.stepReward = -1f / MaxStep;   // эпизод без результата стоит ровно -1

        areaScript.agent = agentGo.transform;
        areaScript.target = target.transform;

        var behavior = agentGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 8;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(4);
        behavior.BehaviorType = BehaviorType.Default;

        var requester = agentGo.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = DecisionPeriod;
        // Действие повторяется между решениями: за один шаг физики шар почти
        // не сдвигается, и решение раз в 5 шагов делает эффект заметным.
        requester.TakeActionsBetweenDecisions = true;

        // ---------- Префаб и K арен ----------
        var areaPrefab = PrefabUtility.SaveAsPrefabAsset(area, Root + "/Prefabs/TrainingArea.prefab");
        Object.DestroyImmediate(area);

        var rootTrainingAreas = new GameObject("TrainingAreas");
        var rootCameras = new GameObject("Cameras");
        var rootLighting = new GameObject("Lighting");
        new GameObject("Managers");
        new GameObject("UI");
        new GameObject("Debug");

        for (int i = 0; i < AreaCount; i++)
        {
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(areaPrefab, rootTrainingAreas.transform);
            instance.name = $"TrainingArea_{i:00}";
            instance.transform.localPosition = new Vector3(i * AreaSpacing, 0f, 0f);
            instance.GetComponent<RollerArea>().areaIndex = i;  // смещение сида арены
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 12f, -14f);
        cam.transform.rotation = Quaternion.Euler(40f, 0f, 0f);
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
        Debug.Log($"RollerBall-сцена собрана: {ScenePath}, арен: {AreaCount}, шаг {AreaSpacing}");
    }

    private static void CreateWall(Transform parent, string name, Vector3 localPosition, Vector3 scale)
    {
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = name;
        wall.tag = "wall";
        wall.transform.SetParent(parent, false);
        wall.transform.localPosition = localPosition;
        wall.transform.localScale = scale;
        wall.GetComponent<Renderer>().sharedMaterial = MakeMaterial("WallMat", ColWall);
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
