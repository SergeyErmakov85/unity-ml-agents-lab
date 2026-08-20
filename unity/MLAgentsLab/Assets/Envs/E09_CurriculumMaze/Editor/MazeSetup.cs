using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E09_CurriculumMaze` по TS-009: сетка до 11 × 11,
/// пул стен на максимальный размер, агент, выход, K тренировочных арен.
/// Меню: Tools → RL → Build Maze Scene.
///
/// Сцена строится кодом целиком: чтобы её изменить, меняется этот скрипт
/// и сцена пересобирается.
/// </summary>
public static class MazeSetup
{
    /// <summary>Идентификатор среды. Behavior Name обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E09_CurriculumMaze";

    private const string Root = "Assets/Envs/E09_CurriculumMaze";
    private const string ScenePath = Root + "/Scenes/E09_CurriculumMaze.unity";

    private const int AreaCount = 8;

    /// <summary>Шаг арен по X. Максимальная сетка 11 × 2 = 22, шаг 30 даёт зазор 8.</summary>
    private const float AreaSpacing = 30f;

    /// <summary>
    /// Лимит времени эпизода в ходах.
    ///
    /// Значение выбрано измерением, а не на глаз. При 600 ходах случайная
    /// политика доходила до выхода в 43 % эпизодов даже на сетке 11 × 11:
    /// время случайного блуждания по сетке из n клеток растёт как n·log n
    /// (≈ 580 ходов при n = 121), и шестисот ходов ей хватало. Задача,
    /// которую решает случайная политика, ничему не учит, а учебный план
    /// на ней не показывает вообще ничего.
    ///
    /// При 150 ходах кратчайший путь (6 ходов на лёгком уровне, ≈ 17 на
    /// тяжёлом) остаётся с запасом в 9–25 раз, а случайное блуждание
    /// перестаёт укладываться в лимит.
    /// </summary>
    private const int MaxStep = 150;

    /// <summary>
    /// Решение на каждом шаге Academy: движение телепортом по клеткам,
    /// промежуточных состояний нет и ждать нечего.
    /// </summary>
    private const int DecisionPeriod = 1;

    // --- лучевой сенсор ---------------------------------------------------
    /// <summary>Лучей в каждую сторону от центрального: 2 -> всего 5 лучей.</summary>
    private const int RaysPerDirection = 2;

    private const float MaxRayDegrees = 90f;
    private const float RayLength = 12f;

    /// <summary>
    /// Два кадра лучей вместо одного. Лабиринт частично наблюдаем: из коридора
    /// видно только коридор, и один кадр не отличает «иду вперёд» от «стою
    /// в тупике». Это самая дешёвая память, и, в отличие от рекуррентной
    /// политики, она не требует выхода `memory_out` в графе ONNX.
    /// </summary>
    private const int ObservationStacks = 2;

    private static readonly string[] DetectableTags = { "wall", "goal" };

    private static readonly Color ColFloor = new Color(0.20f, 0.22f, 0.27f);
    private static readonly Color ColWall = new Color(0.48f, 0.50f, 0.58f);
    private static readonly Color ColBorder = new Color(0.32f, 0.34f, 0.40f);
    private static readonly Color ColAgent = new Color(0.25f, 0.75f, 0.95f);
    private static readonly Color ColGoal = new Color(0.35f, 0.85f, 0.45f);

    [MenuItem("Tools/RL/Build Maze Scene")]
    public static void BuildScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        int max = MazeArea.MaxGridSize;
        float cell = MazeArea.CellSize;
        float fullSpan = max * cell;

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<MazeArea>();

        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.SetParent(area.transform, false);
        // Plane в Unity — 10 × 10 единиц при масштабе 1.
        floor.transform.localScale = new Vector3(fullSpan / 10f, 1f, fullSpan / 10f);
        floor.GetComponent<Renderer>().sharedMaterial = MakeMaterial("FloorMat", ColFloor);

        // ---------- Рамка ----------
        // Позиции и масштабы задаёт MazeArea под текущий размер сетки;
        // здесь важны только имена — по ним арена их находит.
        var border = new GameObject("Border");
        border.transform.SetParent(area.transform, false);
        foreach (var side in new[] { "North", "South", "East", "West" })
            CreateBox(border.transform, side, Vector3.zero, Vector3.one, "BorderMat", ColBorder, "wall");
        areaScript.borderRoot = border.transform;

        // ---------- Пул стен ----------
        // Объекты создаются один раз на максимальный размер сетки и дальше
        // только включаются и выключаются: создавать по 40 кубов каждый эпизод
        // в восьми аренах — тысячи аллокаций в секунду (требование 7.5).
        var walls = new GameObject("Walls");
        walls.transform.SetParent(area.transform, false);
        for (int x = 0; x < max; x++)
        {
            for (int y = 0; y < max; y++)
            {
                // Имя — единственный способ восстановить координату клетки
                // по объекту при заполнении пула в MazeArea.EnsurePool().
                var box = CreateBox(walls.transform, $"Cell_{x}_{y}", Vector3.zero,
                                    new Vector3(cell, cell, cell), "WallMat", ColWall, "wall");
                box.gameObject.SetActive(false);
            }
        }
        areaScript.wallRoot = walls.transform;

        // ---------- Выход ----------
        var goalGo = GameObject.CreatePrimitive(PrimitiveType.Cube);
        goalGo.name = "Goal";
        goalGo.tag = "goal";
        goalGo.transform.SetParent(area.transform, false);
        goalGo.transform.localScale = new Vector3(1.4f, 1.4f, 1.4f);
        goalGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("GoalMat", ColGoal);
        // Выход — ориентир для лучей и точка на сетке, а не препятствие:
        // физический коллайдер мешал бы агенту войти в клетку.
        goalGo.GetComponent<BoxCollider>().isTrigger = true;
        areaScript.goal = goalGo.transform;

        // ---------- Агент ----------
        var agentGo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        agentGo.name = "Agent";
        agentGo.tag = "agent";
        agentGo.transform.SetParent(area.transform, false);
        agentGo.transform.localScale = new Vector3(1f, 0.5f, 1f);
        agentGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("AgentMat", ColAgent);
        // Движение телепортом по клеткам: Rigidbody не нужен, а его
        // недетерминизм только мешал бы воспроизводимости.
        Object.DestroyImmediate(agentGo.GetComponent<CapsuleCollider>());

        var agent = agentGo.AddComponent<MazeAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;
        agent.stepPenalty = -1f / MaxStep;
        areaScript.agent = agentGo.transform;

        // ---------- Сенсоры ----------
        // Имя сенсора влияет на порядок наблюдений: ML-Agents сортирует
        // сенсоры по имени, поэтому "RayPerceptionSensor" идёт перед
        // "VectorSensor_size10" и становится obs_0.
        var rays = agentGo.AddComponent<RayPerceptionSensorComponent3D>();
        rays.SensorName = "RayPerceptionSensor";
        rays.DetectableTags = new System.Collections.Generic.List<string>(DetectableTags);
        rays.RaysPerDirection = RaysPerDirection;
        rays.MaxRayDegrees = MaxRayDegrees;
        rays.RayLength = RayLength;
        rays.SphereCastRadius = 0.4f;
        rays.ObservationStacks = ObservationStacks;
        rays.StartVerticalOffset = 0.5f;
        rays.EndVerticalOffset = 0.5f;

        var behavior = agentGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 10;
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
            instance.GetComponent<MazeArea>().areaIndex = i;  // смещение сида арены
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 26f, -20f);
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

        int rays5 = 2 * RaysPerDirection + 1;
        Debug.Log($"Maze-сцена собрана: {ScenePath}; арен: {AreaCount}, " +
                  $"пул стен: {max}×{max}, лучей: {rays5}, стеков: {ObservationStacks}, " +
                  $"размер лучевого наблюдения: {rays5 * (DetectableTags.Length + 2) * ObservationStacks}");
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
