using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E05_FoodCollector` по TS-005: площадка 20×20,
/// хорошая и плохая еда, агент с сеточным сенсором, K тренировочных арен
/// из одного префаба.
/// Меню: Tools → RL → Build FoodCollector Scene.
/// </summary>
public static class FoodCollectorSetup
{
    /// <summary>Идентификатор среды. Behavior Name в Unity обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E05_FoodCollector";

    private const string Root = "Assets/Envs/E05_FoodCollector";
    private const string ScenePath = Root + "/Scenes/E05_FoodCollector.unity";

    private const int AreaCount = 8;
    private const float HalfSize = 10f;
    private const float AreaSpacing = 24f;

    /// <summary>1500 шагов физики = 300 решений при DecisionPeriod = 5.</summary>
    private const int MaxStep = 1500;

    private const int DecisionPeriod = 5;

    private const int GoodFoodCount = 12;
    private const int BadFoodCount = 6;

    // --- сеточный сенсор --------------------------------------------------
    /// <summary>Клеток по каждой стороне сетки.</summary>
    private const int GridCells = 8;

    /// <summary>Размер клетки, м. 8 клеток по 2 м -> обзор 16×16 вокруг агента.</summary>
    private const float CellSize = 2f;

    /// <summary>Теги, различаемые сеткой. По каналу на тег: наблюдение (2, 8, 8).</summary>
    private static readonly string[] DetectableTags = { "goal", "trap" };

    private static readonly Color ColFloor = new Color(0.20f, 0.22f, 0.26f);
    private static readonly Color ColWall = new Color(0.55f, 0.57f, 0.62f);
    private static readonly Color ColGood = new Color(0.25f, 0.85f, 0.40f);
    private static readonly Color ColBad = new Color(0.90f, 0.30f, 0.30f);
    private static readonly Color ColAgent = new Color(0.25f, 0.55f, 0.95f);

    [MenuItem("Tools/RL/Build FoodCollector Scene")]
    public static void BuildFoodCollectorScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<FoodCollectorArea>();
        areaScript.halfSize = HalfSize;

        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.SetParent(area.transform, false);
        floor.transform.localScale = new Vector3(HalfSize / 5f, 1f, HalfSize / 5f);
        floor.GetComponent<Renderer>().sharedMaterial = MakeMaterial("FloorMat", ColFloor);

        var walls = new GameObject("Walls");
        walls.transform.SetParent(area.transform, false);
        float span = 2f * HalfSize + 0.5f;
        CreateBox(walls.transform, "Wall_North", new Vector3(0f, 1f, HalfSize + 0.25f),
                  new Vector3(span, 2f, 0.5f), "WallMat", ColWall, "wall");
        CreateBox(walls.transform, "Wall_South", new Vector3(0f, 1f, -HalfSize - 0.25f),
                  new Vector3(span, 2f, 0.5f), "WallMat", ColWall, "wall");
        CreateBox(walls.transform, "Wall_East", new Vector3(HalfSize + 0.25f, 1f, 0f),
                  new Vector3(0.5f, 2f, span), "WallMat", ColWall, "wall");
        CreateBox(walls.transform, "Wall_West", new Vector3(-HalfSize - 0.25f, 1f, 0f),
                  new Vector3(0.5f, 2f, span), "WallMat", ColWall, "wall");

        // ---------- Еда ----------
        var foodRoot = new GameObject("Food");
        foodRoot.transform.SetParent(area.transform, false);
        for (int i = 0; i < GoodFoodCount + BadFoodCount; i++)
        {
            bool good = i < GoodFoodCount;
            var food = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            food.name = good ? $"Good_{i:00}" : $"Bad_{i - GoodFoodCount:00}";
            food.tag = good ? "goal" : "trap";
            food.transform.SetParent(foodRoot.transform, false);
            food.transform.localPosition = new Vector3(i % 6 * 2f - 5f, 0.5f, i / 6 * 2f - 5f);
            food.transform.localScale = Vector3.one * 0.8f;
            food.GetComponent<Renderer>().sharedMaterial =
                MakeMaterial(good ? "GoodFoodMat" : "BadFoodMat", good ? ColGood : ColBad);
            // Еда — триггер: она не толкается физикой, а поглощается при касании.
            food.GetComponent<SphereCollider>().isTrigger = true;
        }
        areaScript.foodRoot = foodRoot.transform;

        // ---------- Агент ----------
        var agentGo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        agentGo.name = "Collector";
        agentGo.tag = "agent";
        agentGo.transform.SetParent(area.transform, false);
        agentGo.transform.localPosition = new Vector3(0f, 0.5f, 0f);
        agentGo.transform.localScale = new Vector3(1f, 0.5f, 1f);
        agentGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("AgentMat", ColAgent);

        var body = agentGo.AddComponent<Rigidbody>();
        body.mass = 1f;
        body.linearDamping = 1.5f;
        body.angularDamping = 4f;
        body.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        var agent = agentGo.AddComponent<FoodCollectorAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;
        agent.decisionPeriod = DecisionPeriod;

        areaScript.agent = agentGo.transform;

        // ---------- Сеточный сенсор ----------
        // Имя выбрано так, чтобы сортировка сенсоров по имени
        // (Agent.InitializeSensors) поставила сетку перед вектором:
        // "GridSensor" < "VectorSensor_size6". Порядок определяет,
        // что станет obs_0, а что obs_1.
        var grid = agentGo.AddComponent<GridSensorComponent>();
        grid.SensorName = "GridSensor";
        // Высота клетки не 0.01, а 1: клетка — это объём для Physics.OverlapBox,
        // и слишком тонкий слой промахивается мимо шара, лежащего на полу.
        grid.CellScale = new Vector3(CellSize, 1f, CellSize);
        grid.GridSize = new Vector3Int(GridCells, 1, GridCells);
        grid.DetectableTags = (string[])DetectableTags.Clone();
        grid.RotateWithAgent = true;
        grid.ObservationStacks = 1;
        grid.CompressionType = SensorCompressionType.None;
        // ОБЯЗАТЕЛЬНО. Поле m_ColliderMask сериализуется без значения
        // по умолчанию, то есть равно нулю — «ни одного слоя». Сенсор при
        // этом исправно работает и возвращает пустую сетку: ни ошибки,
        // ни предупреждения (измерено, T-16 в docs/07_TROUBLESHOOTING.md).
        grid.ColliderMask = ~0;

        var behavior = agentGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 6;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        // Гибридное пространство: движение непрерывно, ускорение — переключатель.
        behavior.BrainParameters.ActionSpec = new ActionSpec(2, new[] { 2 });
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
            instance.GetComponent<FoodCollectorArea>().areaIndex = i;
        }

        // ---------- Камера и свет ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 22f, -20f);
        cam.transform.rotation = Quaternion.Euler(45f, 0f, 0f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.10f, 0.11f, 0.14f);

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
        Debug.Log($"FoodCollector-сцена собрана: {ScenePath}, арен: {AreaCount}, " +
                  $"еда: {GoodFoodCount} хорошей и {BadFoodCount} плохой, " +
                  $"сетка: {DetectableTags.Length}×{GridCells}×{GridCells}");
    }

    private static Transform CreateBox(Transform parent, string name, Vector3 localPosition,
                                       Vector3 scale, string materialName, Color color, string tag)
    {
        var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.name = name;
        if (tag != null) box.tag = tag;
        box.transform.SetParent(parent, false);
        box.transform.localPosition = localPosition;
        box.transform.localScale = scale;
        box.GetComponent<Renderer>().sharedMaterial = MakeMaterial(materialName, color);
        return box.transform;
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
