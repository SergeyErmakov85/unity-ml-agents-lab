using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E02_CartPoleUnity` по TS-002: рельс, тележка, шест,
/// границы допустимой зоны, K тренировочных арен из одного префаба.
/// Меню: Tools → RL → Build CartPole Scene.
///
/// Сцена строится кодом целиком: чтобы её изменить, меняется этот скрипт
/// и сцена пересобирается.
/// </summary>
public static class CartPoleSetup
{
    /// <summary>Идентификатор среды. Behavior Name в Unity обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E02_CartPoleUnity";

    private const string Root = "Assets/Envs/E02_CartPoleUnity";
    private const string ScenePath = Root + "/Scenes/E02_CartPoleUnity.unity";

    /// <summary>Число арен = число параллельных сред для Python (требование 8.2).</summary>
    private const int AreaCount = 8;

    /// <summary>Шаг размещения арен по Z. Арена занимает 5.5 единиц по X и 1 по Z.</summary>
    private const float AreaSpacing = 3f;

    /// <summary>
    /// Бюджет эпизода. 200 шагов — определение задачи `CartPole-v0`; при награде
    /// +1 за шаг это даёт максимум 200 и классический порог «решено» — 195.
    /// </summary>
    private const int MaxStep = 200;

    /// <summary>Решение каждый шаг: шаг Academy — это и есть шаг интегрирования (τ = 0.02 с).</summary>
    private const int DecisionPeriod = 1;

    private const float XThreshold = 2.4f;

    private static readonly Color ColRail = new Color(0.30f, 0.30f, 0.36f);
    private static readonly Color ColCart = new Color(0.20f, 0.50f, 0.90f);
    private static readonly Color ColPole = new Color(0.85f, 0.65f, 0.25f);
    private static readonly Color ColLimit = new Color(0.75f, 0.25f, 0.25f);

    [MenuItem("Tools/RL/Build CartPole Scene")]
    public static void BuildCartPoleScene()
    {
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<CartPoleArea>();

        CreateBox(area.transform, "Rail", new Vector3(0f, -0.05f, 0f),
                  new Vector3(2f * XThreshold + 1f, 0.1f, 0.4f), "RailMat", ColRail);

        CreateBox(area.transform, "Limit_Left", new Vector3(-XThreshold, 0.15f, 0f),
                  new Vector3(0.05f, 0.4f, 0.4f), "LimitMat", ColLimit);
        CreateBox(area.transform, "Limit_Right", new Vector3(XThreshold, 0.15f, 0f),
                  new Vector3(0.05f, 0.4f, 0.4f), "LimitMat", ColLimit);

        var cart = CreateBox(area.transform, "Cart", new Vector3(0f, 0.15f, 0f),
                             new Vector3(0.5f, 0.3f, 0.4f), "CartMat", ColCart);

        // Пивот шеста — отдельный пустой объект в точке крепления: вращать надо
        // вокруг основания шеста, а масштабированный куб вращается вокруг центра.
        // Пивот — сосед тележки, а не её потомок: вращение внутри неравномерно
        // отмасштабированного родителя даёт сдвиговую деформацию. Позицию по X
        // пивоту переносит `CartPoleArea.ApplyToVisuals`.
        var pivot = new GameObject("PolePivot");
        pivot.transform.SetParent(area.transform, false);
        pivot.transform.localPosition = new Vector3(0f, 0.3f, 0f);

        // Полная длина шеста — 2·length = 1.0 м; куб растёт вверх от пивота.
        CreateBox(pivot.transform, "Pole", new Vector3(0f, 0.5f, 0f),
                  new Vector3(0.08f, 1.0f, 0.08f), "PoleMat", ColPole);

        areaScript.cart = cart;
        areaScript.polePivot = pivot.transform;
        areaScript.xThreshold = XThreshold;

        var agentGo = new GameObject("Agent");
        agentGo.tag = "agent";
        agentGo.transform.SetParent(area.transform, false);

        var agent = agentGo.AddComponent<CartPoleAgent>();
        agent.area = areaScript;
        agent.MaxStep = MaxStep;

        var behavior = agentGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        behavior.BrainParameters.VectorObservationSize = 4;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(2);
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
            // Арены разносятся по Z: так все тележки видны в кадре одной камерой.
            instance.transform.localPosition = new Vector3(0f, 0f, i * AreaSpacing);
            instance.GetComponent<CartPoleArea>().areaIndex = i;  // смещение сида арены
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 6f, -8f);
        cam.transform.rotation = Quaternion.Euler(20f, 0f, 0f);
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
        Debug.Log($"CartPole-сцена собрана: {ScenePath}, арен: {AreaCount}, MaxStep: {MaxStep}");
    }

    /// <summary>Создаёт куб без коллайдера: физики в этой среде нет.</summary>
    private static Transform CreateBox(Transform parent, string name, Vector3 localPosition,
                                       Vector3 scale, string materialName, Color color)
    {
        var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.name = name;
        box.transform.SetParent(parent, false);
        box.transform.localPosition = localPosition;
        box.transform.localScale = scale;
        box.GetComponent<Renderer>().sharedMaterial = MakeMaterial(materialName, color);
        Object.DestroyImmediate(box.GetComponent<BoxCollider>());
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
