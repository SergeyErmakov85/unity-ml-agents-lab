using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

/// <summary>
/// Одноразовая (идемпотентная) настройка общего проекта лаборатории:
/// создаёт и назначает URP-пайплайн, теги RL-сред и список сцен сборки.
/// Меню: Tools → RL → Configure Project.
/// </summary>
public static class ProjectBootstrap
{
    private const string SettingsFolder = "Assets/Settings";
    private const string RendererPath = SettingsFolder + "/URP_Renderer.asset";
    private const string PipelinePath = SettingsFolder + "/URP_Asset.asset";

    private static readonly string[] ScenePaths =
    {
        "Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/Scenes/RLTrainingScene.unity",
        "Assets/ML-ENVIRONMENTS/02-Examples/Greed_world/Scenes/GridWorld.unity",
    };

    [MenuItem("Tools/RL/Configure Project")]
    public static void Configure()
    {
        EnsureFolder(SettingsFolder);

        var rendererData = AssetDatabase.LoadAssetAtPath<UniversalRendererData>(RendererPath);
        if (rendererData == null)
        {
            rendererData = ScriptableObject.CreateInstance<UniversalRendererData>();
            AssetDatabase.CreateAsset(rendererData, RendererPath);
        }

        var pipeline = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(PipelinePath);
        if (pipeline == null)
        {
            pipeline = UniversalRenderPipelineAsset.Create(rendererData);
            AssetDatabase.CreateAsset(pipeline, PipelinePath);
        }

        GraphicsSettings.defaultRenderPipeline = pipeline;
        QualitySettings.renderPipeline = pipeline;

        foreach (var tag in new[] { "agent", "goal", "trap", "wall" })
        {
            if (System.Array.IndexOf(InternalEditorUtility.tags, tag) < 0)
                InternalEditorUtility.AddTag(tag);
        }

        var scenes = new EditorBuildSettingsScene[ScenePaths.Length];
        for (int i = 0; i < ScenePaths.Length; i++)
            scenes[i] = new EditorBuildSettingsScene(ScenePaths[i], true);
        EditorBuildSettings.scenes = scenes;

        AssetDatabase.SaveAssets();
        Debug.Log("ProjectBootstrap: URP назначен, теги и сцены сборки настроены.");
    }

    /// <summary>Открывает все сцены сред по очереди — ошибки попадут в лог редактора.</summary>
    [MenuItem("Tools/RL/Validate Scenes")]
    public static void ValidateScenes()
    {
        foreach (var path in ScenePaths)
        {
            var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
            Debug.Log($"ProjectBootstrap: сцена открыта без сбоев — {path} (объектов: {scene.rootCount})");
        }
    }

    public static void ConfigureAndValidate()
    {
        Configure();
        ValidateScenes();
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
