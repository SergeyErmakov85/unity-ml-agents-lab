using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Перекрашивает сцену GridWorld под визуальный стиль среды Frozen Lake из Gymnasium.
/// Каждому типу объекта соответствует своя текстура (CC0):
///   ловушки  → Tex_IceHole.jpg   (прорубь: замёрзшая тёмная вода, ambientCG Ice002)
///   стены    → Tex_Snowdrift.jpg (сугроб: мягкий снег, Poly Haven snow_02)
///   цель     → Tex_Grass.jpg     (зелёная трава, ambientCG Grass001)
///   пол      → Tex_IceChecker.png (процедурная шахматка льда)
/// Скрипт меняет материалы, текстуру линий сетки, фон камеры, свет и ambient в сцене,
/// после применения сохраняет предпросмотр в Logs/GridWorldPreview.png.
/// Обратная операция восстанавливает исходную палитру TS-001 (без текстур).
/// Меню: Tools → RL → GridWorld Theme → Frozen Lake / Default (TS-001) / Preview Screenshot.
/// Headless: -executeMethod GridWorldFrozenLakeTheme.ApplyFrozenLake
///           -executeMethod GridWorldFrozenLakeTheme.ApplyDefault
///           -executeMethod GridWorldFrozenLakeTheme.CapturePreview
/// </summary>
public static class GridWorldFrozenLakeTheme
{
    private const string Root = "Assets/ML-ENVIRONMENTS/02-Examples/Greed_world";
    private const string ScenePath = Root + "/Scenes/GridWorld.unity";
    private const string GridTexPath = Root + "/Textures/Tex_GridLines.png";
    private const string IceTexPath = Root + "/Textures/Tex_IceChecker.png";
    private const string HoleTexPath = Root + "/Textures/Tex_IceHole.jpg";
    private const string SnowTexPath = Root + "/Textures/Tex_Snowdrift.jpg";
    private const string GrassTexPath = Root + "/Textures/Tex_Grass.jpg";
    private const string PreviewPath = "Logs/GridWorldPreview.png";

    /// <summary>Полная палитра сцены: материалы (цвет × текстура) + камера + освещение.</summary>
    private struct Palette
    {
        public Color Ground;      // Mat_Ground (тон пола; при текстуре — множитель)
        public Color Start;       // Mat_Start
        public Color Goal;        // Mat_Goal
        public Color Trap;        // Mat_Trap
        public Color Wall;        // Mat_Wall
        public Color Agent;       // Mat_Agent
        public Color GridLines;   // линии сетки (текстура)
        public Color Background;  // фон камеры
        public Color Ambient;     // ambient-свет
        public Color Light;       // directional light
        public bool UseIceChecker; // шахматная текстура льда на полу
        public Color IceCheckerA;
        public Color IceCheckerB;
        public string TrapTexture; // путь к текстуре или null (без текстуры)
        public string WallTexture;
        public string GoalTexture;
    }

    /// <summary>
    /// Палитра Frozen Lake (Gymnasium): лёд, проруби, трава на целевой клетке.
    /// Цвет материала в URP умножается на текстуру, поэтому для ловушки задан
    /// тёмно-синий множитель — светлый лёд Ice002 превращается в тёмную воду проруби.
    /// </summary>
    private static Palette FrozenLake => new Palette
    {
        Ground = Color.white,
        Start = Hex("#EAD9A6"),      // соломенный коврик на старте
        Goal = Color.white,          // трава без тонировки
        Trap = Hex("#5E86AC"),       // притемняет лёд до тёмной воды полыньи
        Wall = Color.white,          // снег без тонировки
        Agent = Hex("#E74C3C"),      // красный костюм эльфа
        GridLines = Hex("#FFFFFF"),  // белые швы между льдинами
        Background = Hex("#0F3A5F"), // глубокая ледниковая синева
        Ambient = Hex("#8FA8BF"),
        Light = Hex("#EAF4FF"),      // холодный дневной свет
        UseIceChecker = true,
        IceCheckerA = Hex("#A8D5EE"),
        IceCheckerB = Hex("#C4E4F5"),
        TrapTexture = HoleTexPath,
        WallTexture = SnowTexPath,
        GoalTexture = GrassTexPath,
    };

    /// <summary>Исходная палитра TS-001 (константы GridWorldSetup), без текстур.</summary>
    private static Palette Default => new Palette
    {
        Ground = Hex("#2B2B33"),
        Start = Hex("#F1C40F"),
        Goal = Hex("#2ECC71"),
        Trap = Hex("#E74C3C"),
        Wall = Hex("#7F8C8D"),
        Agent = Hex("#3498DB"),
        GridLines = Hex("#4A4A57"),
        Background = Hex("#202028"),
        Ambient = Hex("#404050"),
        Light = Hex("#FFF4E5"),
        UseIceChecker = false,
    };

    [MenuItem("Tools/RL/GridWorld Theme/Frozen Lake")]
    public static void ApplyFrozenLake() => Apply(FrozenLake, "Frozen Lake");

    [MenuItem("Tools/RL/GridWorld Theme/Default (TS-001)")]
    public static void ApplyDefault() => Apply(Default, "Default (TS-001)");

    // ---------------------------------------------------------------- Применение

    private static void Apply(Palette palette, string themeName)
    {
        if (!AssetDatabase.LoadAssetAtPath<Material>($"{Root}/Materials/Mat_Ground.mat"))
        {
            Debug.LogError(
                "GridWorld: материалы не найдены. Сначала соберите сцену: " +
                "Tools → RL → Build GridWorld Scene (GridWorldSetup.BuildScene).");
            return;
        }

        RecolorMaterials(palette);
        RegenerateGridLinesTexture(palette.GridLines);
        RetintScene(palette);

        AssetDatabase.SaveAssets();
        Debug.Log($"GridWorld: тема «{themeName}» применена (материалы, текстуры, камера, свет).");

        CapturePreview();
    }

    private static void RecolorMaterials(Palette palette)
    {
        SetMaterial("Mat_Start", palette.Start, null);
        SetMaterial("Mat_Goal", palette.Goal, palette.GoalTexture);
        SetMaterial("Mat_Trap", palette.Trap, palette.TrapTexture);
        SetMaterial("Mat_Wall", palette.Wall, palette.WallTexture);
        SetMaterial("Mat_Agent", palette.Agent, null);

        // Пол: в теме Frozen Lake — шахматная текстура льда, иначе сплошной цвет.
        string groundTex = null;
        if (palette.UseIceChecker)
        {
            MakeIceCheckerTexture(palette.IceCheckerA, palette.IceCheckerB);
            groundTex = IceTexPath;
        }
        SetMaterial("Mat_Ground", palette.Ground, groundTex);
    }

    /// <summary>Задаёт материалу цвет и базовую текстуру (null — убрать текстуру).</summary>
    private static void SetMaterial(string name, Color color, string texturePath)
    {
        var mat = AssetDatabase.LoadAssetAtPath<Material>($"{Root}/Materials/{name}.mat");
        if (mat == null)
        {
            Debug.LogWarning($"GridWorld: материал {name} не найден — пропущен.");
            return;
        }

        Texture2D tex = null;
        if (texturePath != null)
        {
            tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
            if (tex == null)
                Debug.LogWarning($"GridWorld: текстура {texturePath} не найдена — {name} останется без неё.");
        }

        mat.color = color;
        mat.SetTexture("_BaseMap", tex);
        mat.SetTexture("_MainTex", tex);
        EditorUtility.SetDirty(mat);
    }

    /// <summary>Открывает сцену и перекрашивает то, что живёт в ней: камеру, свет, ambient.</summary>
    private static void RetintScene(Palette palette)
    {
        if (!File.Exists(ScenePath))
        {
            Debug.LogWarning($"GridWorld: сцена {ScenePath} не найдена — камера и свет не перекрашены.");
            return;
        }

        var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

        var camGo = GameObject.Find("MainCamera");
        if (camGo != null && camGo.TryGetComponent(out Camera cam))
        {
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = palette.Background;
        }

        var lightGo = GameObject.Find("DirectionalLight");
        if (lightGo != null && lightGo.TryGetComponent(out Light light))
            light.color = palette.Light;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = palette.Ambient;

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
    }

    // ---------------------------------------------------------------- Предпросмотр

    /// <summary>Рендерит сцену GridWorld главной камерой в Logs/GridWorldPreview.png (1024×1024).</summary>
    [MenuItem("Tools/RL/GridWorld Theme/Preview Screenshot")]
    public static void CapturePreview()
    {
        if (!File.Exists(ScenePath))
        {
            Debug.LogWarning($"GridWorld: сцена {ScenePath} не найдена — предпросмотр пропущен.");
            return;
        }

        EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

        var camGo = GameObject.Find("MainCamera");
        if (camGo == null || !camGo.TryGetComponent(out Camera cam))
        {
            Debug.LogWarning("GridWorld: MainCamera не найдена — предпросмотр пропущен.");
            return;
        }

        const int size = 1024;
        var rt = new RenderTexture(size, size, 24);
        var prevTarget = cam.targetTexture;
        var prevActive = RenderTexture.active;

        // Иначе вместо URP Lit рендерится заглушка асинхронной компиляции (однотонный кадр).
        bool prevAsync = ShaderUtil.allowAsyncCompilation;
        ShaderUtil.allowAsyncCompilation = false;

        // URP не поддерживает ручной Camera.Render() — кадр нужно запрашивать у SRP.
        var request = new RenderPipeline.StandardRequest();
        if (RenderPipeline.SupportsRenderRequest(cam, request))
        {
            request.destination = rt;
            RenderPipeline.SubmitRenderRequest(cam, request); // прогрев: компиляция шейдеров
            RenderPipeline.SubmitRenderRequest(cam, request);
        }
        else
        {
            cam.targetTexture = rt;
            cam.Render();
        }
        ShaderUtil.allowAsyncCompilation = prevAsync;

        RenderTexture.active = rt;
        var tex = new Texture2D(size, size, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, size, size), 0, 0);
        tex.Apply();

        cam.targetTexture = prevTarget;
        RenderTexture.active = prevActive;

        Directory.CreateDirectory(Path.GetDirectoryName(PreviewPath));
        File.WriteAllBytes(PreviewPath, tex.EncodeToPNG());
        Object.DestroyImmediate(tex);
        Object.DestroyImmediate(rt);
        Debug.Log($"GridWorld: предпросмотр сохранён: {PreviewPath}");
    }

    // ---------------------------------------------------------------- Текстуры

    /// <summary>
    /// Перегенерирует Tex_GridLines.png с новым цветом линий
    /// (геометрия как в GridWorldSetup: 500×500 px, 5 клеток по 100 px, линии 3 px).
    /// </summary>
    private static void RegenerateGridLinesTexture(Color lineColor)
    {
        const int size = 500;
        const int cellPx = 100;
        const int half = 1; // линия 3 px: центр ± 1

        var pixels = new Color[size * size];
        var clear = new Color(0f, 0f, 0f, 0f);
        for (int i = 0; i < pixels.Length; i++) pixels[i] = clear;

        for (int line = 0; line <= 5; line++)
        {
            int center = Mathf.Clamp(line * cellPx, half, size - 1 - half);
            for (int offset = -half; offset <= half; offset++)
            {
                int pos = center + offset;
                for (int i = 0; i < size; i++)
                {
                    pixels[pos * size + i] = lineColor; // горизонтальная линия
                    pixels[i * size + pos] = lineColor; // вертикальная линия
                }
            }
        }

        WritePng(GridTexPath, size, pixels);
    }

    /// <summary>
    /// Шахматная текстура льда 500×500: клетки 100 px двух чередующихся оттенков,
    /// как чередование льдин в рендере Frozen Lake.
    /// </summary>
    private static void MakeIceCheckerTexture(Color a, Color b)
    {
        const int size = 500;
        const int cellPx = 100;

        var pixels = new Color[size * size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                bool even = ((x / cellPx) + (y / cellPx)) % 2 == 0;
                pixels[y * size + x] = even ? a : b;
            }
        }

        WritePng(IceTexPath, size, pixels);
    }

    private static void WritePng(string path, int size, Color[] pixels)
    {
        var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        tex.SetPixels(pixels);
        tex.Apply();
        File.WriteAllBytes(path, tex.EncodeToPNG());
        Object.DestroyImmediate(tex);

        AssetDatabase.ImportAsset(path);
        var importer = (TextureImporter)AssetImporter.GetAtPath(path);
        importer.wrapMode = TextureWrapMode.Clamp;
        importer.alphaIsTransparency = true;
        importer.SaveAndReimport();
    }

    private static Color Hex(string hex)
    {
        ColorUtility.TryParseHtmlString(hex, out var color);
        return color;
    }
}
