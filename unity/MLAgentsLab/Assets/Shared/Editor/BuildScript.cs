using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace LabRL.EditorTools
{
    /// <summary>
    /// Headless-сборка среды из командной строки (требование 7.6).
    ///
    /// <code>
    /// Unity.exe -batchmode -quit -projectPath &lt;path&gt; \
    ///   -executeMethod LabRL.EditorTools.BuildScript.BuildEnv \
    ///   -envId E03_RollerBall -outputPath ..\..\builds\E03_RollerBall
    /// </code>
    ///
    /// Про код возврата. В batch-режиме Unity завершает процесс с кодом 0, даже
    /// если <c>executeMethod</c> выбросил исключение, — поэтому мы не полагаемся
    /// на исключение, а вызываем <c>EditorApplication.Exit</c> с явным кодом.
    /// Без этого CI считает провалившуюся сборку успешной.
    /// </summary>
    public static class BuildScript
    {
        const string EnvsFolder = "Assets/Envs";

        /// <summary>Точка входа для CLI. Читает -envId и -outputPath из аргументов.</summary>
        public static void BuildEnv()
        {
            try
            {
                string envId = GetArg("-envId");
                string outputPath = GetArg("-outputPath");

                if (string.IsNullOrEmpty(envId))
                    Fail("не задан обязательный аргумент -envId (например -envId E03_RollerBall)");

                if (string.IsNullOrEmpty(outputPath))
                    outputPath = Path.Combine("..", "..", "builds", envId);

                var report = Build(envId, outputPath);
                if (report.summary.result != BuildResult.Succeeded)
                {
                    Fail($"сборка {envId} завершилась со статусом {report.summary.result}, " +
                         $"ошибок: {report.summary.totalErrors}");
                }

                Debug.Log($"BuildScript: {envId} собран за {report.summary.totalTime}, " +
                          $"размер {report.summary.totalSize} байт, путь {report.summary.outputPath}");
                EditorApplication.Exit(0);
            }
            catch (Exception exc)
            {
                Fail($"{exc.GetType().Name}: {exc.Message}\n{exc.StackTrace}");
            }
        }

        /// <summary>
        /// Собирает одну среду. Вынесено из <see cref="BuildEnv"/>, чтобы сборку
        /// можно было вызвать из редактора и из тестов, не завершая процесс.
        /// </summary>
        /// <param name="envId">Идентификатор среды, например <c>E03_RollerBall</c>.</param>
        /// <param name="outputPath">
        /// Каталог сборки. Относительные пути считаются от каталога Unity-проекта,
        /// то есть <c>..\..\builds\X</c> указывает в <c>&lt;repo&gt;\builds\X</c>.
        /// </param>
        public static BuildReport Build(string envId, string outputPath)
        {
            string scenePath = FindScene(envId);
            string exePath = Path.Combine(outputPath, envId + ".exe");
            Directory.CreateDirectory(outputPath);

            var options = new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = exePath,
                target = BuildTarget.StandaloneWindows64,
                targetGroup = BuildTargetGroup.Standalone,
                // Development-сборка не нужна: билд используется только как
                // источник опыта для Python, а профилирование идёт в редакторе.
                options = BuildOptions.None,
            };

            Debug.Log($"BuildScript: собираю {envId}\n  сцена: {scenePath}\n  выход: {exePath}");
            return BuildPipeline.BuildPlayer(options);
        }

        /// <summary>
        /// Находит единственную сцену среды. Неоднозначность — ошибка: сборка
        /// «какой-нибудь» сцены даёт билд, в котором Python не найдёт нужное поведение.
        /// </summary>
        static string FindScene(string envId)
        {
            string folder = $"{EnvsFolder}/{envId}";
            if (!AssetDatabase.IsValidFolder(folder))
                Fail($"каталог среды не найден: {folder}");

            string[] guids = AssetDatabase.FindAssets("t:Scene", new[] { folder });
            string[] scenes = guids.Select(AssetDatabase.GUIDToAssetPath).OrderBy(p => p, StringComparer.Ordinal).ToArray();

            if (scenes.Length == 0)
                Fail($"в {folder} нет ни одной сцены");
            if (scenes.Length > 1)
                Fail($"в {folder} найдено {scenes.Length} сцен ({string.Join(", ", scenes)}); " +
                     "среда обязана содержать ровно одну сцену");

            return scenes[0];
        }

        /// <summary>Читает значение именованного аргумента командной строки.</summary>
        static string GetArg(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == name)
                    return args[i + 1];
            }
            return null;
        }

        /// <summary>Пишет ошибку и завершает процесс с ненулевым кодом (требование 7.6).</summary>
        static void Fail(string message)
        {
            Debug.LogError("BuildScript: " + message);
            Console.Error.WriteLine("BuildScript: " + message);
            EditorApplication.Exit(1);
        }
    }
}
