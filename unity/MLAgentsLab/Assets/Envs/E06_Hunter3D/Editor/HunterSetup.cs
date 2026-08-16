using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E06_Hunter3D` по TS-006: площадка 20×20 со стенами
/// и препятствиями, охотник с лучевым сенсором, цель, K тренировочных арен
/// из одного префаба.
/// Меню: Tools → RL → Build Hunter Scene.
///
/// Сцена строится кодом целиком: чтобы её изменить, меняется этот скрипт
/// и сцена пересобирается.
/// </summary>
public static class HunterSetup
{
    /// <summary>Идентификатор среды. Behavior Name в Unity обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E06_Hunter3D";

    private const string Root = "Assets/Envs/E06_Hunter3D";
    private const string ScenePath = Root + "/Scenes/E06_Hunter3D.unity";

    /// <summary>Число арен = число параллельных сред для Python (требование 8.2).</summary>
    private const int AreaCount = 8;

    /// <summary>Половина стороны площадки: 10 -> площадка 20×20.</summary>
    private const float HalfSize = 10f;

    /// <summary>Шаг размещения арен по X. Площадка занимает 20 единиц, шаг 24 даёт зазор 4.</summary>
    private const float AreaSpacing = 24f;

    private const int MaxStep = 1000;

    /// <summary>
    /// Решение раз в 5 шагов физики. За один шаг физики тяга почти не сдвигает
    /// охотника, и решение на каждом шаге тратило бы инференс впустую.
    /// Эпизод в 1000 шагов Academy даёт 200 решений — столько же шагов MDP.
    /// </summary>
    private const int DecisionPeriod = 5;

    // --- лучевой сенсор ---------------------------------------------------
    /// <summary>Лучей в каждую сторону от центрального: 1 -> всего 3 луча.</summary>
    private const int RaysPerDirection = 1;

    /// <summary>Половина угла конуса: 35° -> конус 70°.</summary>
    private const float MaxRayDegrees = 35f;

    private const float RayLength = 20f;

    /// <summary>Теги, различаемые лучами. Порядок фиксирован ТЗ и влияет на наблюдение.</summary>
    private static readonly string[] DetectableTags = { "wall", "obstacle", "goal" };

    private static readonly Color ColFloor = new Color(0.24f, 0.26f, 0.30f);
    private static readonly Color ColWall = new Color(0.55f, 0.57f, 0.62f);
    private static readonly Color ColObstacle = new Color(0.45f, 0.35f, 0.55f);
    private static readonly Color ColHunter = new Color(0.20f, 0.60f, 0.90f);
    private static readonly Color ColTarget = new Color(0.90f, 0.35f, 0.30f);

    /// <summary>Локальные центры препятствий. Планировка одинакова во всех аренах.</summary>
    private static readonly Vector3[] ObstacleCenters =
    {
        new Vector3(-4.5f, 1f, -4.5f),
        new Vector3(4.5f, 1f, -4.5f),
        new Vector3(-4.5f, 1f, 4.5f),
        new Vector3(4.5f, 1f, 4.5f),
        new Vector3(0f, 1f, 0f),
    };

    [MenuItem("Tools/RL/Build Hunter Scene")]
    public static void BuildHunterScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<HunterArea>();
        areaScript.halfSize = HalfSize;

        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.SetParent(area.transform, false);
        floor.transform.localScale = new Vector3(HalfSize / 5f, 1f, HalfSize / 5f);
        floor.GetComponent<Renderer>().sharedMaterial = MakeMaterial("FloorMat", ColFloor);

        var walls = new GameObject("Walls");
        walls.transform.SetParent(area.transform, false);
        float wallSpan = 2f * HalfSize + 0.5f;
        CreateBox(walls.transform, "Wall_North", new Vector3(0f, 1f, HalfSize + 0.25f),
                  new Vector3(wallSpan, 2f, 0.5f), "WallMat", ColWall, "wall");
        CreateBox(walls.transform, "Wall_South", new Vector3(0f, 1f, -HalfSize - 0.25f),
                  new Vector3(wallSpan, 2f, 0.5f), "WallMat", ColWall, "wall");
        CreateBox(walls.transform, "Wall_East", new Vector3(HalfSize + 0.25f, 1f, 0f),
                  new Vector3(0.5f, 2f, wallSpan), "WallMat", ColWall, "wall");
        CreateBox(walls.transform, "Wall_West", new Vector3(-HalfSize - 0.25f, 1f, 0f),
                  new Vector3(0.5f, 2f, wallSpan), "WallMat", ColWall, "wall");

        var obstacles = new GameObject("Obstacles");
        obstacles.transform.SetParent(area.transform, false);
        for (int i = 0; i < ObstacleCenters.Length; i++)
        {
            CreateBox(obstacles.transform, $"Obstacle_{i}", ObstacleCenters[i],
                      new Vector3(3f, 2f, 3f), "ObstacleMat", ColObstacle, "obstacle");
        }
        areaScript.obstacleRoot = obstacles.transform;

        var targetGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        targetGo.name = "Target";
        targetGo.tag = "goal";
        targetGo.transform.SetParent(area.transform, false);
        targetGo.transform.localPosition = new Vector3(5f, 0.5f, 5f);
        targetGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("TargetMat", ColTarget);
        // Цель — ориентир для лучей и для дистанции, а не физическое препятствие:
        // столкновение с ней сдвинуло бы её и исказило задачу.
        targetGo.GetComponent<SphereCollider>().isTrigger = true;

        var hunterGo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        hunterGo.name = "Hunter";
        hunterGo.tag = "agent";
        hunterGo.transform.SetParent(area.transform, false);
        hunterGo.transform.localPosition = new Vector3(-5f, 0.5f, -5f);
        hunterGo.transform.localScale = new Vector3(1f, 0.5f, 1f);
        hunterGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("HunterMat", ColHunter);

        var body = hunterGo.AddComponent<Rigidbody>();
        body.mass = 1f;
        // Демпфирование заменяет трение: без него охотник бесконечно скользит
        // по инерции, и управление тягой перестаёт быть управлением.
        body.linearDamping = 1.5f;
        body.angularDamping = 4f;
        // Поворот задаётся действием через transform.Rotate, поэтому вращение
        // от физики заморожено — иначе капсула заваливается от толчков о стены.
        body.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        var agent = hunterGo.AddComponent<HunterAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;
        agent.decisionPeriod = DecisionPeriod;
        // Штраф за время: эпизод без результата стоит ровно -1 при
        // MaxStep / DecisionPeriod = 200 решениях.
        agent.stepPenalty = -(float)DecisionPeriod / MaxStep;

        areaScript.hunter = hunterGo.transform;
        areaScript.target = targetGo.transform;

        // ---------- Сенсоры ----------
        // Лучи: «зрение» агента. Имя сенсора влияет на порядок наблюдений —
        // ML-Agents сортирует сенсоры по имени (Agent.InitializeSensors),
        // поэтому "RayPerceptionSensor" идёт перед "VectorSensor_size10"
        // и становится obs_0.
        var rays = hunterGo.AddComponent<RayPerceptionSensorComponent3D>();
        rays.SensorName = "RayPerceptionSensor";
        rays.DetectableTags = new System.Collections.Generic.List<string>(DetectableTags);
        rays.RaysPerDirection = RaysPerDirection;
        rays.MaxRayDegrees = MaxRayDegrees;
        rays.RayLength = RayLength;
        rays.SphereCastRadius = 0.5f;
        rays.ObservationStacks = 1;
        rays.StartVerticalOffset = 0.5f;
        rays.EndVerticalOffset = 0.5f;

        var behavior = hunterGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 10;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
        behavior.BehaviorType = BehaviorType.Default;

        var requester = hunterGo.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = DecisionPeriod;
        // ВАЖНО: действие НЕ повторяется между решениями. `OnActionReceived`
        // должен вызываться ровно раз на решение, потому что решение — это шаг
        // MDP: и формирующая награда, и штраф за время начисляются на шаг MDP,
        // а не на шаг физики (T-15 в docs/07_TROUBLESHOOTING.md).
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
            instance.GetComponent<HunterArea>().areaIndex = i;  // смещение сида арены
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 22f, -20f);
        cam.transform.rotation = Quaternion.Euler(45f, 0f, 0f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.10f, 0.11f, 0.14f);

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
        Debug.Log($"Hunter-сцена собрана: {ScenePath}, арен: {AreaCount}, " +
                  $"лучей: {2 * RaysPerDirection + 1}, тегов: {DetectableTags.Length}, " +
                  $"размер лучевого наблюдения: {(2 * RaysPerDirection + 1) * (DetectableTags.Length + 2)}");
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
