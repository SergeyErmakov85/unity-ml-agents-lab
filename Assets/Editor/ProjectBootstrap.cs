using System.Collections.Generic;
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

    public const string EnvironmentsRoot = "Assets/ML-ENVIRONMENTS";

    /// <summary>
    /// Все сцены сред, найденные под <see cref="EnvironmentsRoot"/>.
    /// Список не захардкожен: новая среда попадает в сборку автоматически.
    /// </summary>
    public static string[] ScenePaths
    {
        get
        {
            var guids = AssetDatabase.FindAssets("t:Scene", new[] { EnvironmentsRoot });
            var paths = new List<string>(guids.Length);
            foreach (var guid in guids)
                paths.Add(AssetDatabase.GUIDToAssetPath(guid));
            paths.Sort(System.StringComparer.Ordinal);
            return paths.ToArray();
        }
    }

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

        var paths = ScenePaths;
        var scenes = new EditorBuildSettingsScene[paths.Length];
        for (int i = 0; i < paths.Length; i++)
            scenes[i] = new EditorBuildSettingsScene(paths[i], true);
        EditorBuildSettings.scenes = scenes;

        AssetDatabase.SaveAssets();
        Debug.Log($"ProjectBootstrap: URP назначен, теги настроены, сцен в сборке — {paths.Length}.");
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

    /// <summary>Точка входа для batch-режима: настройка + проверка сцен + проверка готовности к обучению.</summary>
    public static void ConfigureAndValidate()
    {
        Configure();
        ValidateScenes();
        MLAgentsTrainingValidator.Validate();
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
