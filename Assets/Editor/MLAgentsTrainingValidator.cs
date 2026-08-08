using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Проверяет, что каждая среда в Assets/ML-ENVIRONMENTS готова к запуску
/// обучения через mlagents-learn: у агентов задан Behavior Name, к нему есть
/// trainer-YAML в config/ этой же среды, имена behavior-ов не дублируются
/// между средами, у агента есть источник наблюдений и запросчик решений.
///
/// Меню: Tools → RL → Validate Training Setup
/// Batch: -executeMethod MLAgentsTrainingValidator.Validate
/// </summary>
public static class MLAgentsTrainingValidator
{
    private class BehaviorUsage
    {
        public string Environment;
        public string Source;       // путь к сцене или префабу
        public BehaviorParameters Parameters;
        public bool HasDecisionRequester;
        public bool HasSensorComponent;
    }

    [MenuItem("Tools/RL/Validate Training Setup")]
    public static void Validate()
    {
        // Проверка открывает сцены сред по очереди. Сначала даём сохранить текущую,
        // а в конце возвращаем её обратно — иначе легко потерять несохранённую работу.
        var openedBefore = EditorSceneManager.GetActiveScene().path;
        if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
        {
            Debug.Log("Проверка готовности к обучению отменена пользователем.");
            return;
        }

        try
        {
            ValidateCore();
        }
        finally
        {
            if (!string.IsNullOrEmpty(openedBefore) &&
                EditorSceneManager.GetActiveScene().path != openedBefore)
            {
                EditorSceneManager.OpenScene(openedBefore, OpenSceneMode.Single);
            }
        }
    }

    private static void ValidateCore()
    {
        var errors = new List<string>();
        var warnings = new List<string>();
        var report = new StringBuilder("Готовность сред к обучению\n");

        // behavior name -> среды, где он встречается (для поиска дублей)
        var behaviorOwners = new Dictionary<string, HashSet<string>>();

        foreach (var envPath in EnumerateEnvironments())
        {
            var envName = Path.GetFileName(envPath);
            var usages = CollectBehaviors(envPath, envName);
            var configured = CollectConfiguredBehaviors(envPath, out var configFiles);

            if (usages.Count == 0)
            {
                // Папка без агентов — заготовка среды, не ошибка.
                if (configured.Count > 0)
                    warnings.Add($"{envName}: есть config/*.yaml ({string.Join(", ", configured)}), но ни одного Agent с Behavior Parameters не найдено.");
                continue;
            }

            report.AppendLine($"\n  {envPath}");
            report.AppendLine($"    конфиги: {(configFiles.Count == 0 ? "—" : string.Join(", ", configFiles.Select(Path.GetFileName)))}");

            foreach (var usage in usages)
            {
                var bp = usage.Parameters;
                var name = bp.BehaviorName;
                var where = $"{envName} → {usage.Source} → {bp.gameObject.name}";

                if (string.IsNullOrWhiteSpace(name))
                {
                    errors.Add($"{where}: пустой Behavior Name — mlagents-learn не сможет сопоставить агента с конфигом.");
                    continue;
                }

                if (!behaviorOwners.TryGetValue(name, out var owners))
                    behaviorOwners[name] = owners = new HashSet<string>();
                owners.Add(envName);

                var inConfig = configured.Contains(name);
                report.AppendLine($"    {name,-24} obs={DescribeObservations(bp)}  act={DescribeActions(bp)}  " +
                                  $"config={(inConfig ? "да" : "НЕТ")}  ({usage.Source})");

                if (!inConfig)
                {
                    errors.Add($"{where}: behavior \"{name}\" не описан ни в одном {envName}/config/*.yaml. " +
                               "Добавьте секцию behaviors: → " + name + ":");
                }

                if (!usage.HasDecisionRequester)
                {
                    warnings.Add($"{where}: нет компонента Decision Requester. " +
                                 "Если агент не вызывает RequestDecision() из кода, решения запрашиваться не будут.");
                }

                if (bp.BrainParameters.VectorObservationSize == 0 && !usage.HasSensorComponent)
                {
                    warnings.Add($"{where}: Vector Observation Size = 0 и нет ни одного Sensor Component — агент ничего не наблюдает.");
                }

                var spec = bp.BrainParameters.ActionSpec;
                if (spec.NumContinuousActions == 0 && (spec.BranchSizes == null || spec.BranchSizes.Length == 0))
                {
                    errors.Add($"{where}: не задано ни одного действия (Continuous Actions = 0, Discrete Branches = 0).");
                }
            }
        }

        foreach (var pair in behaviorOwners.Where(p => p.Value.Count > 1))
        {
            errors.Add($"Behavior \"{pair.Key}\" используется в нескольких средах ({string.Join(", ", pair.Value)}). " +
                       "Имена behavior-ов должны быть уникальны в пределах проекта.");
        }

        Debug.Log(report.ToString());
        foreach (var w in warnings) Debug.LogWarning("Training setup: " + w);
        foreach (var e in errors) Debug.LogError("Training setup: " + e);

        var summary = $"Проверка готовности к обучению: ошибок — {errors.Count}, предупреждений — {warnings.Count}.";
        if (errors.Count > 0)
        {
            // В batch-режиме исключение из -executeMethod даёт ненулевой код возврата.
            throw new BuildFailedExceptionLite(summary);
        }
        Debug.Log(summary);
    }

    /// <summary>Assets/ML-ENVIRONMENTS/&lt;NN-Категория&gt;/&lt;Среда&gt;</summary>
    private static IEnumerable<string> EnumerateEnvironments()
    {
        var root = ProjectBootstrap.EnvironmentsRoot;
        if (!AssetDatabase.IsValidFolder(root)) yield break;

        foreach (var category in AssetDatabase.GetSubFolders(root).OrderBy(p => p, System.StringComparer.Ordinal))
            foreach (var env in AssetDatabase.GetSubFolders(category).OrderBy(p => p, System.StringComparer.Ordinal))
                yield return env;
    }

    private static List<BehaviorUsage> CollectBehaviors(string envPath, string envName)
    {
        var result = new List<BehaviorUsage>();
        var search = new[] { envPath };

        foreach (var guid in AssetDatabase.FindAssets("t:Prefab", search))
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (prefab == null) continue;
            foreach (var bp in prefab.GetComponentsInChildren<BehaviorParameters>(true))
                result.Add(Describe(bp, envName, path));
        }

        foreach (var guid in AssetDatabase.FindAssets("t:Scene", search))
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
            foreach (var root in scene.GetRootGameObjects())
                foreach (var bp in root.GetComponentsInChildren<BehaviorParameters>(true))
                    result.Add(Describe(bp, envName, path));
        }

        // Один и тот же behavior встречается и в префабе, и в сцене — оставляем по одному на имя.
        return result
            .GroupBy(u => u.Parameters.BehaviorName)
            .Select(g => g.First())
            .ToList();
    }

    private static BehaviorUsage Describe(BehaviorParameters bp, string envName, string source)
    {
        var go = bp.gameObject;
        return new BehaviorUsage
        {
            Environment = envName,
            Source = Path.GetFileName(source),
            Parameters = bp,
            HasDecisionRequester = go.GetComponent<DecisionRequester>() != null,
            HasSensorComponent = go.GetComponentsInChildren<SensorComponent>(true).Length > 0,
        };
    }

    /// <summary>Имена behavior-ов из всех config/*.yaml среды (ключи под `behaviors:`).</summary>
    private static HashSet<string> CollectConfiguredBehaviors(string envPath, out List<string> configFiles)
    {
        var names = new HashSet<string>();
        configFiles = new List<string>();

        var dir = Path.Combine(envPath, "config");
        if (!Directory.Exists(dir)) return names;

        foreach (var file in Directory.GetFiles(dir, "*.y*ml", SearchOption.AllDirectories))
        {
            configFiles.Add(file);
            var inBehaviors = false;
            foreach (var raw in File.ReadAllLines(file))
            {
                var line = raw.TrimEnd();
                if (line.Length == 0 || line.TrimStart().StartsWith("#")) continue;

                if (!line.StartsWith(" ") && !line.StartsWith("\t"))
                {
                    inBehaviors = line.StartsWith("behaviors:");
                    continue;
                }
                if (!inBehaviors) continue;

                // Ровно два пробела отступа + имя + двоеточие.
                var indent = line.Length - line.TrimStart(' ').Length;
                if (indent != 2) continue;
                var trimmed = line.Trim();
                var colon = trimmed.IndexOf(':');
                if (colon > 0 && trimmed.Substring(colon + 1).Trim().Length == 0)
                    names.Add(trimmed.Substring(0, colon).Trim());
            }
        }
        return names;
    }

    private static string DescribeObservations(BehaviorParameters bp)
    {
        var p = bp.BrainParameters;
        var stacked = p.NumStackedVectorObservations > 1 ? $"×{p.NumStackedVectorObservations}" : "";
        var sensors = bp.GetComponentsInChildren<SensorComponent>(true).Length;
        return sensors > 0 ? $"{p.VectorObservationSize}{stacked}+{sensors}sens" : $"{p.VectorObservationSize}{stacked}";
    }

    private static string DescribeActions(BehaviorParameters bp)
    {
        var spec = bp.BrainParameters.ActionSpec;
        var parts = new List<string>();
        if (spec.NumContinuousActions > 0) parts.Add($"cont{spec.NumContinuousActions}");
        if (spec.BranchSizes != null && spec.BranchSizes.Length > 0)
            parts.Add("disc[" + string.Join(",", spec.BranchSizes) + "]");
        return parts.Count == 0 ? "—" : string.Join("+", parts);
    }

    /// <summary>Отдельный тип, чтобы отличать сбой валидации от прочих исключений в логе.</summary>
    private class BuildFailedExceptionLite : System.Exception
    {
        public BuildFailedExceptionLite(string message) : base(message) { }
    }
}
