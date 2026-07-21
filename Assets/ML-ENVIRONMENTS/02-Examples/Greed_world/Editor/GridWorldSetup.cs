using System.IO;
using TMPro;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

/// <summary>
/// Собирает сцену GridWorld по TS-001: сетка 5×5, агент, цель, ловушки, стены,
/// UI-панель статистики, ортографическая камера и освещение.
/// Меню: Tools → RL → Build GridWorld Scene.
/// </summary>
public static class GridWorldSetup
{
    private const string Root = "Assets/ML-ENVIRONMENTS/02-Examples/Greed_world";
    private const string ScenePath = Root + "/Scenes/GridWorld.unity";

    private static readonly Color ColBackground = Hex("#202028");
    private static readonly Color ColGround = Hex("#2B2B33");
    private static readonly Color ColGridLines = Hex("#4A4A57");
    private static readonly Color ColStart = Hex("#F1C40F");
    private static readonly Color ColGoal = Hex("#2ECC71");
    private static readonly Color ColTrap = Hex("#E74C3C");
    private static readonly Color ColWall = Hex("#7F8C8D");
    private static readonly Color ColAgent = Hex("#3498DB");
    private static readonly Color ColText = Hex("#ECF0F1");
    private static readonly Color ColAmbient = Hex("#404050");
    private static readonly Color ColLight = Hex("#FFF4E5");

    [MenuItem("Tools/RL/Build GridWorld Scene")]
    public static void BuildScene()
    {
        EnsureFolders();
        EnsureTags("agent", "goal", "trap", "wall");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Префабы ловушки и стены ----------
        var trapPrefab = BuildTrapPrefab();
        var wallPrefab = BuildWallPrefab();

        // ---------- TrainingArea ----------
        var area = new GameObject("TrainingArea_01");
        var env = area.AddComponent<GridWorldEnvironment>();

        // Ground
        var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
        ground.name = "Ground";
        ground.transform.SetParent(area.transform, false);
        ground.transform.localPosition = new Vector3(0f, -0.05f, 0f);
        ground.transform.localScale = new Vector3(5f, 0.1f, 5f);
        ground.GetComponent<Renderer>().sharedMaterial =
            MakeLitMaterial("Mat_Ground", ColGround);

        // GridLines
        var gridLines = GameObject.CreatePrimitive(PrimitiveType.Quad);
        gridLines.name = "GridLines";
        gridLines.transform.SetParent(area.transform, false);
        gridLines.transform.localPosition = new Vector3(0f, 0.011f, 0f);
        gridLines.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        gridLines.transform.localScale = new Vector3(5f, 5f, 1f);
        Object.DestroyImmediate(gridLines.GetComponent<Collider>());
        gridLines.GetComponent<Renderer>().sharedMaterial = MakeGridLinesMaterial();

        // Markers
        var markers = new GameObject("Markers");
        markers.transform.SetParent(area.transform, false);

        var startMarker = MakeFlatMarker("StartMarker", markers.transform,
            new Vector3(-2f, 0.011f, -2f), MakeLitMaterial("Mat_Start", ColStart));
        Object.DestroyImmediate(startMarker.GetComponent<Collider>());

        var goalMarker = MakeFlatMarker("GoalMarker", markers.transform,
            new Vector3(2f, 0.011f, 2f), MakeLitMaterial("Mat_Goal", ColGoal));
        goalMarker.GetComponent<BoxCollider>().isTrigger = true;
        goalMarker.tag = "goal";

        InstantiatePrefab(trapPrefab, "TrapMarker_01", markers.transform, new Vector3(-1f, 0.011f, 1f));
        InstantiatePrefab(trapPrefab, "TrapMarker_02", markers.transform, new Vector3(1f, 0.011f, -1f));

        // Walls
        var walls = new GameObject("Walls");
        walls.transform.SetParent(area.transform, false);
        InstantiatePrefab(wallPrefab, "Wall_01", walls.transform, new Vector3(-1f, 0.5f, 0f));
        InstantiatePrefab(wallPrefab, "Wall_02", walls.transform, new Vector3(1f, 0.5f, 1f));
        InstantiatePrefab(wallPrefab, "Wall_03", walls.transform, new Vector3(1f, 0.5f, 2f));

        // ---------- Агент ----------
        var agentGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        agentGo.name = "Agent";
        agentGo.tag = "agent";
        agentGo.transform.SetParent(area.transform, false);
        agentGo.transform.localPosition = new Vector3(-2f, 0.3f, -2f);
        agentGo.transform.localScale = new Vector3(0.6f, 0.6f, 0.6f);
        agentGo.GetComponent<Renderer>().sharedMaterial =
            MakeLitMaterial("Mat_Agent", ColAgent);

        var agent = agentGo.AddComponent<GridWorldAgent>();
        agent.env = env;
        agent.MaxStep = env.maxSteps;

        var behavior = agentGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = "GridWorldQLearning";
        behavior.BrainParameters.VectorObservationSize = 25;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(4);
        behavior.BehaviorType = BehaviorType.Default;

        var requester = agentGo.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = 1;
        requester.TakeActionsBetweenDecisions = false;

        // ---------- Префаб TrainingArea ----------
        PrefabUtility.SaveAsPrefabAssetAndConnect(
            area, Root + "/Prefabs/TrainingArea.prefab", InteractionMode.AutomatedAction);

        // ---------- UI ----------
        BuildUI(agent);

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.orthographic = true;
        cam.orthographicSize = 3f;
        cam.transform.position = new Vector3(0f, 10f, 0f);
        cam.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = ColBackground;

        // ---------- Освещение ----------
        var lightGo = GameObject.Find("Directional Light");
        lightGo.name = "DirectionalLight";
        var light = lightGo.GetComponent<Light>();
        light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        light.intensity = 1f;
        light.color = ColLight;
        light.shadows = LightShadows.Soft;
        light.shadowStrength = 0.6f;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = ColAmbient;
        RenderSettings.ambientIntensity = 1f;

        // ---------- Сохранение ----------
        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();
        Debug.Log($"GridWorld-сцена собрана и сохранена: {ScenePath}");
    }

    // ---------------------------------------------------------------- Префабы

    private static GameObject BuildTrapPrefab()
    {
        var trap = GameObject.CreatePrimitive(PrimitiveType.Cube);
        trap.name = "TrapMarker";
        trap.tag = "trap";
        trap.transform.localScale = new Vector3(0.9f, 0.02f, 0.9f);
        trap.GetComponent<BoxCollider>().isTrigger = true;
        trap.GetComponent<Renderer>().sharedMaterial = MakeLitMaterial("Mat_Trap", ColTrap);
        var prefab = PrefabUtility.SaveAsPrefabAsset(trap, Root + "/Prefabs/TrapMarker.prefab");
        Object.DestroyImmediate(trap);
        return prefab;
    }

    private static GameObject BuildWallPrefab()
    {
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = "Wall";
        wall.tag = "wall";
        wall.GetComponent<Renderer>().sharedMaterial = MakeLitMaterial("Mat_Wall", ColWall);
        var prefab = PrefabUtility.SaveAsPrefabAsset(wall, Root + "/Prefabs/Wall.prefab");
        Object.DestroyImmediate(wall);
        return prefab;
    }

    private static void InstantiatePrefab(GameObject prefab, string name, Transform parent, Vector3 localPos)
    {
        var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab, parent);
        go.name = name;
        go.transform.localPosition = localPos;
    }

    private static GameObject MakeFlatMarker(string name, Transform parent, Vector3 localPos, Material mat)
    {
        var marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
        marker.name = name;
        marker.transform.SetParent(parent, false);
        marker.transform.localPosition = localPos;
        marker.transform.localScale = new Vector3(0.9f, 0.02f, 0.9f);
        marker.GetComponent<Renderer>().sharedMaterial = mat;
        return marker;
    }

    // ---------------------------------------------------------------- UI

    private static void BuildUI(GridWorldAgent agent)
    {
        var canvasGo = new GameObject("UICanvas");
        var canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;

        var scaler = canvasGo.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920f, 1080f);
        scaler.matchWidthOrHeight = 0.5f;
        canvasGo.AddComponent<GraphicRaycaster>();

        var panelGo = new GameObject("StatsPanel");
        panelGo.transform.SetParent(canvasGo.transform, false);
        var panelRect = panelGo.AddComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0f, 1f);
        panelRect.anchorMax = new Vector2(0f, 1f);
        panelRect.pivot = new Vector2(0f, 1f);
        panelRect.anchoredPosition = new Vector2(20f, -20f);
        panelRect.sizeDelta = new Vector2(320f, 200f);
        var panelImage = panelGo.AddComponent<Image>();
        panelImage.color = new Color(ColBackground.r, ColBackground.g, ColBackground.b, 0.6f);

        var ui = canvasGo.AddComponent<GridWorldUI>();
        ui.agent = agent;
        ui.textEpisode = MakeText(panelRect, "Text_Episode", 0, "Episode: 1");
        ui.textStep = MakeText(panelRect, "Text_Step", 1, "Step: 0 / 100");
        ui.textState = MakeText(panelRect, "Text_State", 2, "State: 0");
        ui.textLastAction = MakeText(panelRect, "Text_LastAction", 3, "Action: -");
        ui.textReward = MakeText(panelRect, "Text_Reward", 4, "Reward: 0.00");
        ui.textResult = MakeText(panelRect, "Text_Result", 5, "Result: Running");
    }

    private static TextMeshProUGUI MakeText(RectTransform parent, string name, int row, string initial)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        var rect = go.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(0f, 1f);
        rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = new Vector2(12f, -6f - row * 32f);
        rect.sizeDelta = new Vector2(296f, 30f);

        var text = go.AddComponent<TextMeshProUGUI>();
        text.text = initial;
        text.fontSize = 22f;
        text.color = ColText;
        text.alignment = TextAlignmentOptions.Left;
        return text;
    }

    // ---------------------------------------------------------------- Ассеты

    private static void EnsureFolders()
    {
        foreach (var folder in new[] { "Materials", "Prefabs", "Scenes", "Textures" })
            EnsureFolder($"{Root}/{folder}");
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

    private static void EnsureTags(params string[] tags)
    {
        foreach (var tag in tags)
        {
            if (System.Array.IndexOf(InternalEditorUtility.tags, tag) < 0)
                InternalEditorUtility.AddTag(tag);
        }
    }

    private static Material MakeLitMaterial(string name, Color color)
    {
        string path = $"{Root}/Materials/{name}.mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            mat = new Material(shader) { color = color };
            mat.SetFloat("_Metallic", 0f);
            mat.SetFloat("_Smoothness", 0.1f);
            AssetDatabase.CreateAsset(mat, path);
        }
        return mat;
    }

    /// <summary>Unlit-прозрачный материал с процедурной текстурой линий сетки 5×5.</summary>
    private static Material MakeGridLinesMaterial()
    {
        string matPath = Root + "/Materials/Mat_GridLines.mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
        if (mat != null) return mat;

        var tex = MakeGridTexture();

        var shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) shader = Shader.Find("Unlit/Transparent");
        mat = new Material(shader);
        mat.SetTexture("_BaseMap", tex);
        mat.SetTexture("_MainTex", tex);
        mat.SetColor("_BaseColor", Color.white);
        // URP Unlit → Surface Type = Transparent (alpha blend)
        mat.SetFloat("_Surface", 1f);
        mat.SetFloat("_Blend", 0f);
        mat.SetOverrideTag("RenderType", "Transparent");
        mat.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        mat.renderQueue = (int)RenderQueue.Transparent;
        AssetDatabase.CreateAsset(mat, matPath);
        return mat;
    }

    /// <summary>Текстура 500×500: 5 клеток по 100 px, линии толщиной 3 px (0.03 юнита).</summary>
    private static Texture2D MakeGridTexture()
    {
        const string texPath = Root + "/Textures/Tex_GridLines.png";
        var existing = AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
        if (existing != null) return existing;

        const int size = 500;
        const int cellPx = 100;
        const int half = 1; // линия 3 px: центр ± 1

        var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        var clear = new Color(0f, 0f, 0f, 0f);
        var pixels = new Color[size * size];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = clear;

        for (int line = 0; line <= 5; line++)
        {
            int center = Mathf.Clamp(line * cellPx, half, size - 1 - half);
            for (int offset = -half; offset <= half; offset++)
            {
                int pos = center + offset;
                for (int i = 0; i < size; i++)
                {
                    pixels[pos * size + i] = ColGridLines; // горизонтальная линия
                    pixels[i * size + pos] = ColGridLines; // вертикальная линия
                }
            }
        }

        tex.SetPixels(pixels);
        tex.Apply();
        File.WriteAllBytes(texPath, tex.EncodeToPNG());
        Object.DestroyImmediate(tex);

        AssetDatabase.ImportAsset(texPath);
        var importer = (TextureImporter)AssetImporter.GetAtPath(texPath);
        importer.wrapMode = TextureWrapMode.Clamp;
        importer.alphaIsTransparency = true;
        importer.SaveAndReimport();
        return AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
    }

    private static Color Hex(string hex)
    {
        ColorUtility.TryParseHtmlString(hex, out var color);
        return color;
    }
}
