using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E07_RacingCar` по TS-007: замкнутый кольцевой трек
/// из сегментов стен, восемь контрольных точек, машина с веером лучей,
/// K тренировочных арен из одного префаба.
/// Меню: Tools → RL → Build Racing Scene.
///
/// Сцена строится кодом целиком: чтобы её изменить, меняется этот скрипт
/// и сцена пересобирается.
/// </summary>
public static class RacingSetup
{
    /// <summary>Идентификатор среды. Behavior Name в Unity обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E07_RacingCar";

    private const string Root = "Assets/Envs/E07_RacingCar";
    private const string ScenePath = Root + "/Scenes/E07_RacingCar.unity";

    /// <summary>Число арен = число параллельных сред для Python (требование 8.2).</summary>
    private const int AreaCount = 8;

    /// <summary>Шаг размещения арен по X. Трек занимает 29 единиц, шаг 34 даёт зазор 5.</summary>
    private const float AreaSpacing = 34f;

    private const float TrackRadius = 11.5f;
    private const float TrackWidth = 6f;

    /// <summary>Сегментов в каждой стене. 32 даёт заметно гладкое кольцо.</summary>
    private const int WallSegments = 32;

    private const int CheckpointCount = 8;

    /// <summary>
    /// Бюджет эпизода: 2000 шагов физики = 400 решений. Круга длиной 72 м
    /// на средней скорости 10 м/с хватает примерно за 150 решений; запас
    /// нужен ранним, медленным заездам.
    /// </summary>
    private const int MaxStep = 2000;

    private const int DecisionPeriod = 5;

    // --- лучевой сенсор ---------------------------------------------------
    /// <summary>Лучей в каждую сторону: 4 -> всего 9 лучей (проект-3 курса).</summary>
    private const int RaysPerDirection = 4;

    /// <summary>Половина угла веера: 90° -> обзор 180°.</summary>
    private const float MaxRayDegrees = 90f;

    private const float RayLength = 20f;

    /// <summary>Луч различает только стены: контрольные точки — не препятствие.</summary>
    private static readonly string[] DetectableTags = { "wall" };

    /// <summary>
    /// Встроенный слой Unity «Ignore Raycast». Контрольные точки кладутся
    /// на него, чтобы лучи их не видели: иначе агент принимал бы триггер
    /// за стену поперёк трассы.
    /// </summary>
    private const int IgnoreRaycastLayer = 2;

    private static readonly Color ColGround = new Color(0.16f, 0.17f, 0.20f);
    private static readonly Color ColTrack = new Color(0.28f, 0.29f, 0.33f);
    private static readonly Color ColWall = new Color(0.60f, 0.62f, 0.68f);
    private static readonly Color ColCar = new Color(0.95f, 0.75f, 0.20f);
    private static readonly Color ColCheckpoint = new Color(0.20f, 0.85f, 0.55f);

    [MenuItem("Tools/RL/Build Racing Scene")]
    public static void BuildRacingScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        float outerRadius = TrackRadius + TrackWidth / 2f;
        float innerRadius = TrackRadius - TrackWidth / 2f;

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<RacingArea>();
        areaScript.trackRadius = TrackRadius;
        areaScript.trackWidth = TrackWidth;

        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.SetParent(area.transform, false);
        ground.transform.localScale = new Vector3((outerRadius + 2f) / 5f, 1f, (outerRadius + 2f) / 5f);
        ground.GetComponent<Renderer>().sharedMaterial = MakeMaterial("GroundMat", ColGround);

        // Полотно трека — плоское кольцо из плиток. Чисто визуальный ориентир:
        // границы задают стены, а не покрытие.
        var surface = new GameObject("TrackSurface");
        surface.transform.SetParent(area.transform, false);
        for (int i = 0; i < WallSegments; i++)
        {
            float angle = 2f * Mathf.PI * i / WallSegments;
            var tile = CreateBox(surface.transform, $"Tile_{i:00}",
                                 new Vector3(Mathf.Cos(angle) * TrackRadius, 0.01f, Mathf.Sin(angle) * TrackRadius),
                                 new Vector3(TrackWidth, 0.02f, 2f * Mathf.PI * TrackRadius / WallSegments + 0.2f),
                                 "TrackMat", ColTrack, null);
            tile.localRotation = Quaternion.Euler(0f, -angle * Mathf.Rad2Deg, 0f);
            Object.DestroyImmediate(tile.GetComponent<BoxCollider>());
        }

        var walls = new GameObject("Walls");
        walls.transform.SetParent(area.transform, false);
        BuildWallRing(walls.transform, outerRadius, "Outer");
        BuildWallRing(walls.transform, innerRadius, "Inner");

        // ---------- Контрольные точки ----------
        var checkpoints = new GameObject("Checkpoints");
        checkpoints.transform.SetParent(area.transform, false);
        for (int i = 0; i < CheckpointCount; i++)
        {
            // Точка 0 стоит не на старте, а на 1/8 круга впереди: иначе машина
            // засчитывала бы её в первый же шаг, ещё не тронувшись.
            float angle = 2f * Mathf.PI * (i + 1) / CheckpointCount;
            var point = CreateBox(checkpoints.transform, $"Checkpoint_{i}",
                                  new Vector3(Mathf.Cos(angle) * TrackRadius, 1f, Mathf.Sin(angle) * TrackRadius),
                                  new Vector3(TrackWidth, 2f, 0.3f), "CheckpointMat", ColCheckpoint, "goal");
            point.localRotation = Quaternion.Euler(0f, -angle * Mathf.Rad2Deg, 0f);
            point.GetComponent<BoxCollider>().isTrigger = true;
            // Лучи не должны видеть контрольную точку: встроенный слой
            // «Ignore Raycast» исключён из маски сенсора по умолчанию
            // (RayPerceptionSensorComponentBase.k_PhysicsDefaultLayers = -5).
            point.gameObject.layer = IgnoreRaycastLayer;
            point.GetComponent<Renderer>().enabled = i == 0;  // виден только створ старта
        }
        areaScript.checkpointRoot = checkpoints.transform;

        // ---------- Машина ----------
        var carGo = GameObject.CreatePrimitive(PrimitiveType.Cube);
        carGo.name = "Car";
        carGo.tag = "agent";
        carGo.transform.SetParent(area.transform, false);
        carGo.transform.localPosition = new Vector3(TrackRadius, 0.5f, 0f);
        carGo.transform.localScale = new Vector3(1f, 0.6f, 2f);
        carGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("CarMat", ColCar);

        var body = carGo.AddComponent<Rigidbody>();
        body.mass = 1f;
        // Скорость задаётся действием напрямую, поэтому демпфирование не нужно:
        // оно только спорило бы с заданной скоростью.
        body.linearDamping = 0f;
        body.angularDamping = 0f;
        body.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        var agent = carGo.AddComponent<RacingAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;
        agent.decisionPeriod = DecisionPeriod;

        areaScript.car = carGo.transform;

        var rays = carGo.AddComponent<RayPerceptionSensorComponent3D>();
        rays.SensorName = "RayPerceptionSensor";
        rays.DetectableTags = new System.Collections.Generic.List<string>(DetectableTags);
        rays.RaysPerDirection = RaysPerDirection;
        rays.MaxRayDegrees = MaxRayDegrees;
        rays.RayLength = RayLength;
        rays.SphereCastRadius = 0.3f;
        rays.ObservationStacks = 1;
        rays.StartVerticalOffset = 0.3f;
        rays.EndVerticalOffset = 0.3f;

        var behavior = carGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 8;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
        behavior.BehaviorType = BehaviorType.Default;

        var requester = carGo.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = DecisionPeriod;
        // Действие не повторяется между решениями: решение — это шаг MDP,
        // и награда обязана начисляться ровно раз на шаг MDP
        // (T-15 в docs/07_TROUBLESHOOTING.md).
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
            instance.GetComponent<RacingArea>().areaIndex = i;  // смещение сида арены
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 28f, -24f);
        cam.transform.rotation = Quaternion.Euler(48f, 0f, 0f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.09f, 0.10f, 0.13f);

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
        Debug.Log($"Racing-сцена собрана: {ScenePath}, арен: {AreaCount}, " +
                  $"контрольных точек: {CheckpointCount}, лучей: {2 * RaysPerDirection + 1}, " +
                  $"размер лучевого наблюдения: {(2 * RaysPerDirection + 1) * (DetectableTags.Length + 2)}");
    }

    /// <summary>Строит кольцо стены из сегментов, поставленных по касательной.</summary>
    private static void BuildWallRing(Transform parent, float radius, string prefix)
    {
        float segmentLength = 2f * Mathf.PI * radius / WallSegments + 0.3f;
        for (int i = 0; i < WallSegments; i++)
        {
            float angle = 2f * Mathf.PI * i / WallSegments;
            var segment = CreateBox(parent, $"{prefix}_{i:00}",
                                    new Vector3(Mathf.Cos(angle) * radius, 1f, Mathf.Sin(angle) * radius),
                                    new Vector3(0.4f, 2f, segmentLength), "WallMat", ColWall, "wall");
            // Длинная сторона сегмента — по касательной к окружности.
            segment.localRotation = Quaternion.Euler(0f, -angle * Mathf.Rad2Deg, 0f);
        }
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
