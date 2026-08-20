using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using LabRL.Core;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;
using UnityEngine;

namespace LabRL.EditorTools
{
    /// <summary>
    /// Проверка сцены перед сборкой (требование 7.7).
    ///
    /// Что проверяется:
    /// * Behavior Name каждого агента совпадает с идентификатором среды (правило 5.3);
    /// * размерность наблюдений совпадает с заявленной в <c>ENV_SPEC.md</c>;
    /// * пространство действий совпадает с заявленным в <c>ENV_SPEC.md</c>;
    /// * <c>MaxStep</c> задан явно (требование 7.2);
    /// * на агенте есть <c>DecisionRequester</c>;
    /// * агентов вне <c>TrainingArea</c> нет;
    /// * теги RL (<c>agent</c>, <c>goal</c>, <c>trap</c>, <c>wall</c>) существуют в проекте.
    ///
    /// Как ENV_SPEC.md становится проверяемым. Спецификация — текст для человека,
    /// поэтому в неё добавляется одна машиночитаемая строка-контракт:
    /// <code>
    /// &lt;!-- validator: obs_size=25; discrete_branches=4; continuous_size=0; max_step=100 --&gt;
    /// </code>
    /// Только эти значения и сверяются: остальное — предмет ревью человеком.
    /// Если строки нет, среда получает предупреждение, а не ошибку: спецификации
    /// старых сред дописываются по мере работы (формат зафиксирован в
    /// docs/03_CONVENTIONS.md).
    /// </summary>
    public static class SceneValidator
    {
        const string EnvsFolder = "Assets/Envs";
        static readonly string[] RequiredTags =
        {
            "agent", "goal", "trap", "wall", "obstacle",
            // Теги E08_SoccerArena: порядок Detectable Tags у двух команд
            // зеркальный, и пропажа хотя бы одного тега незаметно ломает
            // перспективу наблюдения (ENV_SPEC.md среды, §4).
            "ball", "playerWest", "playerEast", "goalWest", "goalEast",
        };

        /// <summary>Одна найденная проблема.</summary>
        public readonly struct Issue
        {
            public readonly string EnvId;
            public readonly string Message;
            public readonly bool IsError;

            public Issue(string envId, string message, bool isError)
            {
                EnvId = envId;
                Message = message;
                IsError = isError;
            }

            public override string ToString() => $"[{(IsError ? "ОШИБКА" : "предупреждение")}] {EnvId}: {Message}";
        }

        [MenuItem("Tools/RL/Validate Scenes (SceneValidator)")]
        public static void ValidateAllMenu()
        {
            var issues = ValidateAll();
            if (issues.Count == 0)
            {
                Debug.Log("SceneValidator: замечаний нет.");
                return;
            }

            foreach (var issue in issues)
            {
                if (issue.IsError) Debug.LogError("SceneValidator: " + issue);
                else Debug.LogWarning("SceneValidator: " + issue);
            }
        }

        /// <summary>CLI-точка: ненулевой код возврата при наличии ошибок.</summary>
        public static void ValidateAllBatch()
        {
            var issues = ValidateAll();
            foreach (var issue in issues)
                Console.WriteLine("SceneValidator: " + issue);

            int errors = issues.Count(i => i.IsError);
            Console.WriteLine($"SceneValidator: ошибок {errors}, предупреждений {issues.Count - errors}");
            EditorApplication.Exit(errors > 0 ? 1 : 0);
        }

        /// <summary>Проверяет все среды в <c>Assets/Envs</c>.</summary>
        public static List<Issue> ValidateAll()
        {
            var issues = new List<Issue>();

            foreach (string tag in RequiredTags)
            {
                if (Array.IndexOf(InternalEditorUtility.tags, tag) < 0)
                    issues.Add(new Issue("<проект>", $"тег \"{tag}\" не создан; выполните Tools/RL/Configure Project", true));
            }

            foreach (string envFolder in AssetDatabase.GetSubFolders(EnvsFolder))
                issues.AddRange(ValidateEnv(Path.GetFileName(envFolder), envFolder));

            return issues;
        }

        /// <summary>Проверяет одну среду: открывает её сцену и сверяет с ENV_SPEC.md.</summary>
        public static List<Issue> ValidateEnv(string envId, string envFolder)
        {
            var issues = new List<Issue>();

            string[] scenes = AssetDatabase.FindAssets("t:Scene", new[] { envFolder })
                .Select(AssetDatabase.GUIDToAssetPath)
                .OrderBy(p => p, StringComparer.Ordinal)
                .ToArray();

            if (scenes.Length != 1)
            {
                issues.Add(new Issue(envId, $"ожидалась ровно одна сцена, найдено {scenes.Length}", true));
                if (scenes.Length == 0) return issues;
            }

            var spec = EnvSpec.TryLoad(envFolder);
            if (spec == null)
                issues.Add(new Issue(envId, "в ENV_SPEC.md нет строки-контракта <!-- validator: ... -->; " +
                                            "размерности не проверены", false));

            EditorSceneManager.OpenScene(scenes[0], OpenSceneMode.Single);

            var agents = UnityEngine.Object.FindObjectsByType<Agent>(FindObjectsInactive.Include);
            if (agents.Length == 0)
                issues.Add(new Issue(envId, "в сцене нет ни одного агента", true));

            foreach (var agent in agents)
            {
                string who = agent.gameObject.name;

                var behavior = agent.GetComponent<BehaviorParameters>();
                if (behavior == null)
                {
                    issues.Add(new Issue(envId, $"{who}: нет компонента BehaviorParameters", true));
                    continue;
                }

                if (behavior.BehaviorName != envId)
                    issues.Add(new Issue(envId, $"{who}: Behavior Name = \"{behavior.BehaviorName}\", " +
                                                $"ожидался \"{envId}\" (правило 5.3)", true));

                if (agent.GetComponent<DecisionRequester>() == null)
                    issues.Add(new Issue(envId, $"{who}: нет компонента DecisionRequester", true));

                if (agent.MaxStep <= 0)
                    issues.Add(new Issue(envId, $"{who}: MaxStep = {agent.MaxStep}; " +
                                                "требование 7.2 предписывает задавать его явно", true));

                if (agent.GetComponentInParent<TrainingAreaBase>() == null)
                    issues.Add(new Issue(envId, $"{who}: агент вне TrainingArea " +
                                                "(нет TrainingAreaBase среди родителей)", true));

                if (spec != null)
                    issues.AddRange(CompareWithSpec(envId, who, behavior, agent, spec));
            }

            return issues;
        }

        static IEnumerable<Issue> CompareWithSpec(string envId, string who, BehaviorParameters behavior, Agent agent, EnvSpec spec)
        {
            var brain = behavior.BrainParameters;

            if (spec.ObsSize.HasValue && brain.VectorObservationSize != spec.ObsSize.Value)
            {
                yield return new Issue(envId, $"{who}: VectorObservationSize = {brain.VectorObservationSize}, " +
                                              $"в ENV_SPEC.md заявлено {spec.ObsSize.Value}", true);
            }

            if (spec.ContinuousSize.HasValue && brain.ActionSpec.NumContinuousActions != spec.ContinuousSize.Value)
            {
                yield return new Issue(envId, $"{who}: непрерывных действий {brain.ActionSpec.NumContinuousActions}, " +
                                              $"в ENV_SPEC.md заявлено {spec.ContinuousSize.Value}", true);
            }

            if (spec.DiscreteBranches != null)
            {
                var actual = brain.ActionSpec.BranchSizes ?? Array.Empty<int>();
                if (!actual.SequenceEqual(spec.DiscreteBranches))
                {
                    yield return new Issue(envId, $"{who}: дискретные ветки [{string.Join(",", actual)}], " +
                                                  $"в ENV_SPEC.md заявлено [{string.Join(",", spec.DiscreteBranches)}]", true);
                }
            }

            if (spec.MaxStep.HasValue && agent.MaxStep != spec.MaxStep.Value)
            {
                yield return new Issue(envId, $"{who}: MaxStep = {agent.MaxStep}, " +
                                              $"в ENV_SPEC.md заявлено {spec.MaxStep.Value}", true);
            }
        }

        /// <summary>Машиночитаемая часть ENV_SPEC.md.</summary>
        sealed class EnvSpec
        {
            public int? ObsSize;
            public int? ContinuousSize;
            public int? MaxStep;
            public int[] DiscreteBranches;

            static readonly Regex ContractLine = new Regex(@"<!--\s*validator:(?<body>[^>]*?)-->", RegexOptions.Compiled);

            public static EnvSpec TryLoad(string envFolder)
            {
                string path = Path.Combine(envFolder, "ENV_SPEC.md");
                if (!File.Exists(path)) return null;

                var match = ContractLine.Match(File.ReadAllText(path));
                if (!match.Success) return null;

                var spec = new EnvSpec();
                foreach (string part in match.Groups["body"].Value.Split(';'))
                {
                    string[] kv = part.Split('=');
                    if (kv.Length != 2) continue;

                    string key = kv[0].Trim();
                    string value = kv[1].Trim();

                    switch (key)
                    {
                        case "obs_size":
                            spec.ObsSize = ParseInt(value);
                            break;
                        case "continuous_size":
                            spec.ContinuousSize = ParseInt(value);
                            break;
                        case "max_step":
                            spec.MaxStep = ParseInt(value);
                            break;
                        case "discrete_branches":
                            spec.DiscreteBranches = value.Length == 0
                                ? Array.Empty<int>()
                                : value.Split(',').Select(v => ParseInt(v.Trim()) ?? 0).ToArray();
                            break;
                    }
                }
                return spec;
            }

            static int? ParseInt(string value) =>
                int.TryParse(value, out int parsed) ? parsed : (int?)null;
        }
    }
}
