using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E10_Imitation` по TS-010: фиксированный змеевидный
/// коридор 9 × 9, агент, выход, K тренировочных арен.
/// Меню: Tools → RL → Build Corridor Scene.
///
/// Раскладку стен скрипт **спрашивает** у <see cref="CorridorArea.IsWallCell"/>,
/// а не хранит свою копию: две копии правила рано или поздно разойдутся,
/// и агент будет наблюдать одну планировку, а видеть другую.
/// </summary>
public static class CorridorSetup
{
    /// <summary>Идентификатор среды. Behavior Name обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E10_Imitation";

    private const string Root = "Assets/Envs/E10_Imitation";
    private const string ScenePath = Root + "/Scenes/E10_Imitation.unity";

    private const int AreaCount = 8;

    /// <summary>Шаг арен по X. Сетка 9 × 2 = 18 единиц, шаг 24 даёт зазор 6.</summary>
    private const float AreaSpacing = 24f;

    /// <summary>
    /// Лимит времени эпизода в ходах: 2.5 кратчайших пути (48 ходов).
    /// Запас достаточен для неидеальной политики и недостаточен для
    /// случайного блуждания — ради этого среда и существует (ТЗ §1).
    /// </summary>
    private const int MaxStep = 120;

    /// <summary>Движение телепортом: промежуточных состояний нет, ждать нечего.</summary>
    private const int DecisionPeriod = 1;

    private static readonly Color ColFloor = new Color(0.19f, 0.21f, 0.26f);
    private static readonly Color ColWall = new Color(0.46f, 0.44f, 0.52f);
    private static readonly Color ColBorder = new Color(0.30f, 0.30f, 0.36f);
    private static readonly Color ColAgent = new Color(0.95f, 0.75f, 0.25f);
    private static readonly Color ColGoal = new Color(0.35f, 0.85f, 0.45f);

    [MenuItem("Tools/RL/Build Corridor Scene")]
    public static void BuildScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        int size = CorridorArea.GridSize;
        float cell = CorridorArea.CellSize;
        float span = size * cell;

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<CorridorArea>();

        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.SetParent(area.transform, false);
        // Plane в Unity — 10 × 10 единиц при масштабе 1.
        floor.transform.localScale = new Vector3(span / 10f, 1f, span / 10f);
        floor.GetComponent<Renderer>().sharedMaterial = MakeMaterial("FloorMat", ColFloor);

        // ---------- Рамка ----------
        var border = new GameObject("Border");
        border.transform.SetParent(area.transform, false);
        float half = span * 0.5f;
        CreateBox(border.transform, "North", new Vector3(0f, 1f, half + 0.5f * cell),
                  new Vector3(span + cell, 2f, cell), "BorderMat", ColBorder, "wall");
        CreateBox(border.transform, "South", new Vector3(0f, 1f, -half - 0.5f * cell),
                  new Vector3(span + cell, 2f, cell), "BorderMat", ColBorder, "wall");
        CreateBox(border.transform, "East", new Vector3(half + 0.5f * cell, 1f, 0f),
                  new Vector3(cell, 2f, span + cell), "BorderMat", ColBorder, "wall");
        CreateBox(border.transform, "West", new Vector3(-half - 0.5f * cell, 1f, 0f),
                  new Vector3(cell, 2f, span + cell), "BorderMat", ColBorder, "wall");

        // ---------- Стены коридора ----------
        // Планировка фиксирована, поэтому стены создаются один раз и больше
        // не трогаются: пул, как в E09, здесь не нужен.
        var walls = new GameObject("Walls");
        walls.transform.SetParent(area.transform, false);
        int wallCount = 0;
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                if (!CorridorArea.IsWallCell(x, y)) continue;
                CreateBox(walls.transform, $"Cell_{x}_{y}",
                          CorridorArea.CellToLocal(new Vector2Int(x, y), 1f),
                          new Vector3(cell, cell, cell), "WallMat", ColWall, "wall");
                wallCount++;
            }
        }

        // ---------- Выход ----------
        var goalGo = GameObject.CreatePrimitive(PrimitiveType.Cube);
        goalGo.name = "Goal";
        goalGo.tag = "goal";
        goalGo.transform.SetParent(area.transform, false);
        goalGo.transform.localPosition = CorridorArea.CellToLocal(CorridorArea.GoalCell, 0.5f);
        goalGo.transform.localScale = new Vector3(1.4f, 1.4f, 1.4f);
        goalGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("GoalMat", ColGoal);
        // Выход — ориентир, а не препятствие: коллайдер мешал бы войти в клетку.
        goalGo.GetComponent<BoxCollider>().isTrigger = true;
        areaScript.goal = goalGo.transform;

        // ---------- Агент ----------
        var agentGo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        agentGo.name = "Agent";
        agentGo.tag = "agent";
        agentGo.transform.SetParent(area.transform, false);
        agentGo.transform.localPosition = CorridorArea.CellToLocal(CorridorArea.StartCell, 0.5f);
        agentGo.transform.localScale = new Vector3(1f, 0.5f, 1f);
        agentGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("AgentMat", ColAgent);
        Object.DestroyImmediate(agentGo.GetComponent<CapsuleCollider>());

        var agent = agentGo.AddComponent<CorridorAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;
        agent.stepPenalty = -1f / MaxStep;
        areaScript.agent = agentGo.transform;

        var behavior = agentGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 13;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(4);
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
            instance.GetComponent<CorridorArea>().areaIndex = i;
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 24f, -18f);
        cam.transform.rotation = Quaternion.Euler(55f, 0f, 0f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.10f, 0.11f, 0.14f);

        // ---------- Освещение ----------
        var lightGo = GameObject.Find("Directional Light");
        lightGo.name = "DirectionalLight";
        lightGo.transform.SetParent(rootLighting.transform, true);
        var light = lightGo.GetComponent<Light>();
        light.transform.rotation = Quaternion.Euler(55f, -25f, 0f);
        light.intensity = 1f;
        light.shadows = LightShadows.Soft;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.25f, 0.25f, 0.31f);

        EditorSceneManager.SaveScene(scene, ScenePath);
        AssetDatabase.SaveAssets();

        Debug.Log($"Corridor-сцена собрана: {ScenePath}; арен: {AreaCount}, " +
                  $"сетка: {size}×{size}, стен: {wallCount}, " +
                  $"кратчайший путь: {CorridorArea.ShortestPath} ходов, MaxStep: {MaxStep}");
    }

    private static Transform CreateBox(Transform parent, string name, Vector3 localPosition,
                                       Vector3 scale, string materialName, Color color, string tag)
    {
        var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.name = name;
        box.tag = tag;
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
