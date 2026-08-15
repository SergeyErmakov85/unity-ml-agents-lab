using System;
using System.IO;
using System.Linq;
using LabRL.Core;
using LabRL.DebugTools;
using Unity.InferenceEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LabRL.EditorTools
{
    /// <summary>
    /// Сборка билда для проверки инференса ONNX в Unity (требование 10.6).
    ///
    /// Что делает: берёт сцену среды, назначает агентам обученную модель,
    /// переводит поведение в <c>Inference Only</c>, добавляет
    /// <see cref="InferenceProbe"/> и собирает отдельный билд.
    ///
    /// <code>
    /// Unity.exe -batchmode -quit -projectPath &lt;path&gt; \
    ///   -executeMethod LabRL.EditorTools.InferenceBuild.Build \
    ///   -envId E01_GridWorld -modelPath Assets/Envs/E01_GridWorld/Models/x.onnx \
    ///   -outputPath ..\..\builds\E01_GridWorld_inference
    /// </code>
    ///
    /// Почему отдельный билд, а не Play Mode из batch-режима. Вход в Play Mode
    /// через <c>-executeMethod</c> асинхронен: метод возвращает управление
    /// раньше, чем сцена отработает, и результат приходится ловить в глобальном
    /// состоянии редактора. Отдельный билд даёт то же самое обычным процессом
    /// с обычным кодом возврата и полностью повторяет условия боевого запуска.
    ///
    /// Почему сцена среды не меняется. Изменённая копия сохраняется во временную
    /// сцену вне <c>Assets/Envs</c>: там ровно одна сцена на среду — это правило
    /// проверяют <see cref="BuildScript"/> и <see cref="SceneValidator"/>.
    /// </summary>
    public static class InferenceBuild
    {
        const string EnvsFolder = "Assets/Envs";
        const string TempFolder = "Assets/Shared/Temp";
        const string TempScene = TempFolder + "/InferenceCheck.unity";

        public static void Build()
        {
            try
            {
                string envId = GetArg("-envId");
                string modelPath = GetArg("-modelPath");
                string outputPath = GetArg("-outputPath");

                if (string.IsNullOrEmpty(envId)) Fail("не задан -envId");
                if (string.IsNullOrEmpty(outputPath))
                    outputPath = Path.Combine("..", "..", "builds", envId + "_inference");

                var report = BuildInference(envId, modelPath, outputPath);
                if (report.summary.result != BuildResult.Succeeded)
                    Fail($"сборка проверки инференса {envId} завершилась со статусом {report.summary.result}");

                Debug.Log($"InferenceBuild: {envId} собран, путь {report.summary.outputPath}");
                EditorApplication.Exit(0);
            }
            catch (Exception exc)
            {
                Fail($"{exc.GetType().Name}: {exc.Message}\n{exc.StackTrace}");
            }
        }

        /// <summary>Собирает инференс-билд среды. Возвращает отчёт сборки.</summary>
        /// <param name="envId">Идентификатор среды.</param>
        /// <param name="modelPath">
        /// Путь к <c>.onnx</c> относительно проекта. Пусто — берётся единственная
        /// модель из <c>Assets/Envs/&lt;envId&gt;/Models</c>.
        /// </param>
        /// <param name="outputPath">Каталог сборки.</param>
        public static BuildReport BuildInference(string envId, string modelPath, string outputPath)
        {
            string scenePath = FindScene(envId);
            var model = LoadModel(envId, modelPath);

            EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);

            var agents = UnityEngine.Object.FindObjectsByType<Agent>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            if (agents.Length == 0) Fail($"в сцене {scenePath} нет агентов");

            foreach (var agent in agents)
            {
                var behavior = agent.GetComponent<BehaviorParameters>();
                if (behavior == null) Fail($"{agent.name}: нет BehaviorParameters");

                behavior.Model = model;
                behavior.BehaviorType = BehaviorType.InferenceOnly;
                // Детерминированный инференс: Unity читает deterministic_* выходы.
                // Именно детерминированную политику оценивал Python, и сравнивать
                // надо одно с одним (10.6).
                behavior.DeterministicInference = true;

                // Без этих двух вызовов присвоение НЕ попадёт в сохранённую сцену.
                // Агенты живут внутри инстансов префаба TrainingArea, а правка
                // поля инстанса из кода не создаёт override автоматически:
                // сцена сохранится с исходными значениями, билд получит
                // BehaviorType.Default без модели и молча уйдёт на HeuristicPolicy —
                // агент будет стоять на месте, а эпизоды истекать по MaxStep.
                // Подробности: docs/07_TROUBLESHOOTING.md, T-8.
                PrefabUtility.RecordPrefabInstancePropertyModifications(behavior);
                EditorUtility.SetDirty(behavior);
            }

            var probeHost = new GameObject("InferenceProbe");
            probeHost.AddComponent<InferenceProbe>();

            EnsureFolder(TempFolder);
            var scene = SceneManager.GetActiveScene();
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, TempScene);

            VerifySaved(model);

            Directory.CreateDirectory(outputPath);
            string exePath = Path.Combine(outputPath, envId + "_inference.exe");

            var options = new BuildPlayerOptions
            {
                scenes = new[] { TempScene },
                locationPathName = exePath,
                target = BuildTarget.StandaloneWindows64,
                targetGroup = BuildTargetGroup.Standalone,
                options = BuildOptions.None,
            };

            Debug.Log($"InferenceBuild: собираю {envId}\n  модель: {AssetDatabase.GetAssetPath(model)}\n  выход: {exePath}");
            var report = BuildPipeline.BuildPlayer(options);

            // Временная сцена удаляется всегда: оставшийся файл сбил бы
            // SceneValidator при следующем прогоне.
            AssetDatabase.DeleteAsset(TempScene);
            AssetDatabase.Refresh();

            return report;
        }

        /// <summary>
        /// Перечитывает сохранённую сцену и убеждается, что модель и тип поведения
        /// действительно записаны.
        ///
        /// Проверка не избыточна: несохранённое присвоение не даёт ни ошибки,
        /// ни предупреждения — билд собирается успешно и ведёт себя как сцена
        /// без модели. Такую поломку видно только по результату прогона, поэтому
        /// она ловится здесь, до шестиминутной сборки.
        /// </summary>
        static void VerifySaved(ModelAsset expected)
        {
            EditorSceneManager.OpenScene(TempScene, OpenSceneMode.Single);

            var agents = UnityEngine.Object.FindObjectsByType<Agent>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            foreach (var agent in agents)
            {
                var behavior = agent.GetComponent<BehaviorParameters>();
                if (behavior.Model != expected)
                    Fail($"{agent.name}: модель не сохранилась в сцену " +
                         $"(в сцене {(behavior.Model == null ? "null" : behavior.Model.name)}, ожидалась {expected.name})");
                if (behavior.BehaviorType != BehaviorType.InferenceOnly)
                    Fail($"{agent.name}: BehaviorType не сохранился (в сцене {behavior.BehaviorType})");
            }

            Debug.Log($"InferenceBuild: проверено — модель и Inference Only записаны у {agents.Length} агентов");
        }

        static ModelAsset LoadModel(string envId, string modelPath)
        {
            if (!string.IsNullOrEmpty(modelPath))
            {
                var explicitModel = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelPath);
                if (explicitModel == null) Fail($"модель не найдена или не импортирована: {modelPath}");
                return explicitModel;
            }

            string folder = $"{EnvsFolder}/{envId}/Models";
            if (!AssetDatabase.IsValidFolder(folder))
                Fail($"нет каталога моделей {folder}; сначала обучите агента и экспортируйте ONNX");

            string[] guids = AssetDatabase.FindAssets("t:ModelAsset", new[] { folder });
            if (guids.Length == 0) Fail($"в {folder} нет ни одной импортированной модели");
            if (guids.Length > 1)
                Fail($"в {folder} найдено {guids.Length} моделей; укажите нужную аргументом -modelPath");

            return AssetDatabase.LoadAssetAtPath<ModelAsset>(AssetDatabase.GUIDToAssetPath(guids[0]));
        }

        static string FindScene(string envId)
        {
            string folder = $"{EnvsFolder}/{envId}";
            if (!AssetDatabase.IsValidFolder(folder)) Fail($"каталог среды не найден: {folder}");

            string[] scenes = AssetDatabase.FindAssets("t:Scene", new[] { folder })
                .Select(AssetDatabase.GUIDToAssetPath)
                .OrderBy(p => p, StringComparer.Ordinal)
                .ToArray();

            if (scenes.Length != 1) Fail($"в {folder} ожидалась ровно одна сцена, найдено {scenes.Length}");
            return scenes[0];
        }

        static void EnsureFolder(string path)
        {
            var parts = path.Split('/');
            string current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = $"{current}/{parts[i]}";
                if (!AssetDatabase.IsValidFolder(next)) AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }

        static string GetArg(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == name) return args[i + 1];
            return null;
        }

        static void Fail(string message)
        {
            Debug.LogError("InferenceBuild: " + message);
            Console.Error.WriteLine("InferenceBuild: " + message);
            EditorApplication.Exit(1);
        }
    }
}
