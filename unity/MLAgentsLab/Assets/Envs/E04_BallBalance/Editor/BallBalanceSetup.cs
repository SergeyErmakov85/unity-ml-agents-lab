using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E04_BallBalance` по TS-004: платформа, шар,
/// K тренировочных арен из одного префаба.
/// Меню: Tools → RL → Build BallBalance Scene.
///
/// Сцена строится кодом целиком: чтобы её изменить, меняется этот скрипт
/// и сцена пересобирается.
/// </summary>
public static class BallBalanceSetup
{
    /// <summary>Идентификатор среды. Behavior Name в Unity обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E04_BallBalance";

    private const string Root = "Assets/Envs/E04_BallBalance";
    private const string ScenePath = Root + "/Scenes/E04_BallBalance.unity";

    /// <summary>Число арен = число параллельных сред для Python (требование 8.2).</summary>
    private const int AreaCount = 8;

    /// <summary>Шаг размещения арен по X. Платформа занимает 5 единиц, шаг 8 даёт зазор 3.</summary>
    private const float AreaSpacing = 8f;

    /// <summary>
    /// Бюджет эпизода: 1000 шагов Academy по 0.02 с — 20 секунд удержания.
    /// При награде +0.1 за шаг максимум эпизода равен ровно 100.
    /// </summary>
    private const int MaxStep = 1000;

    /// <summary>Решение каждый шаг физики: наклон должен успевать за шаром.</summary>
    private const int DecisionPeriod = 1;

    private static readonly Color ColPlatform = new Color(0.30f, 0.45f, 0.65f);
    private static readonly Color ColBall = new Color(0.90f, 0.55f, 0.20f);

    [MenuItem("Tools/RL/Build BallBalance Scene")]
    public static void BuildBallBalanceScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<BallBalanceArea>();

        var platform = GameObject.CreatePrimitive(PrimitiveType.Cube);
        platform.name = "Platform";
        platform.tag = "agent";
        platform.transform.SetParent(area.transform, false);
        platform.transform.localPosition = Vector3.zero;
        platform.transform.localScale = new Vector3(5f, 0.5f, 5f);
        platform.GetComponent<Renderer>().sharedMaterial = MakeMaterial("PlatformMat", ColPlatform);

        var ballGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        ballGo.name = "Ball";
        ballGo.transform.SetParent(area.transform, false);
        ballGo.transform.localPosition = new Vector3(0f, 4f, 0f);
        ballGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("BallMat", ColBall);

        var body = ballGo.AddComponent<Rigidbody>();
        body.mass = 1f;
        // Непрерывное определение столкновений: шар быстрый и тонкую платформу
        // при дискретном режиме способен проскочить насквозь.
        body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

        // Агент — компонент самой платформы: её наклон и есть действие.
        var agent = platform.AddComponent<BallBalanceAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;

        areaScript.platform = platform.transform;
        areaScript.ball = body;

        var behavior = platform.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 8;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
        behavior.BehaviorType = BehaviorType.Default;

        var requester = platform.AddComponent<DecisionRequester>();
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
            instance.GetComponent<BallBalanceArea>().areaIndex = i;  // смещение сида арены
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 8f, -12f);
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
        Debug.Log($"BallBalance-сцена собрана: {ScenePath}, арен: {AreaCount}, MaxStep: {MaxStep}");
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
