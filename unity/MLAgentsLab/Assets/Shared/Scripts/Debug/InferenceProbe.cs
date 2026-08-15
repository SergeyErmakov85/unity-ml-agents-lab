using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using LabRL.Core;
using Unity.MLAgents;
using UnityEngine;

namespace LabRL.DebugTools
{
    /// <summary>
    /// Измеряет качество ONNX-модели **внутри Unity** (требование 10.6).
    ///
    /// Зачем это нужно. Python-верификация ONNX (`labrl.export.onnx_verify`)
    /// доказывает, что файл соответствует контракту и численно совпадает
    /// с PyTorch. Она **не** доказывает, что агент в Unity ведёт себя так же:
    /// расходиться могут порядок наблюдений, нормализация, частота принятия
    /// решений. Единственный способ это проверить — дать модель Unity
    /// и сравнить награду с Python-оценкой. Порог приёмки — 0.8 (10.6).
    ///
    /// Проб собирает награды завершённых эпизодов через
    /// <see cref="AgentBase.EpisodeFinished"/>, пишет результат в JSON и
    /// завершает приложение. Аргументы командной строки:
    ///
    /// <code>
    /// E01_GridWorld.exe -batchmode -nographics \
    ///   -inferenceEpisodes 20 -inferenceResult C:\path\result.json
    /// </code>
    ///
    /// Без аргумента <c>-inferenceEpisodes</c> компонент отключает сам себя,
    /// поэтому его присутствие в сцене безвредно для обычных прогонов.
    /// </summary>
    public class InferenceProbe : MonoBehaviour
    {
        [Tooltip("Сколько эпизодов собрать. Переопределяется аргументом -inferenceEpisodes.")]
        public int episodes = 20;

        [Tooltip("Куда писать JSON. Переопределяется аргументом -inferenceResult.")]
        public string resultPath = "inference_result.json";

        [Tooltip("Предохранитель: сколько секунд ждать, прежде чем сдаться и записать частичный результат.")]
        public float timeoutSeconds = 300f;

        readonly List<float> m_Rewards = new List<float>();
        readonly Dictionary<string, int> m_Outcomes = new Dictionary<string, int>();

        bool m_Enabled;
        bool m_Finished;
        float m_StartedAt;

        void Awake()
        {
            ParseCommandLine();

            if (!m_Enabled)
            {
                enabled = false;
                return;
            }

            m_StartedAt = Time.realtimeSinceStartup;
            Debug.Log($"InferenceProbe: собираю {episodes} эпизодов, результат -> {resultPath}");
        }

        void OnEnable()
        {
            if (m_Enabled) AgentBase.EpisodeFinished += OnEpisodeFinished;
        }

        void OnDisable()
        {
            AgentBase.EpisodeFinished -= OnEpisodeFinished;
        }

        float m_NextDiagnosticAt;

        void Update()
        {
            if (!m_Enabled || m_Finished) return;

            // Диагностика раз в 5 секунд. Без неё «эпизоды не набираются»
            // выглядит одинаково для десятка разных причин: нет агентов,
            // Academy не проинициализирована, шаги не идут, эпизоды не кончаются.
            if (Time.realtimeSinceStartup >= m_NextDiagnosticAt)
            {
                m_NextDiagnosticAt = Time.realtimeSinceStartup + 5f;
                LogDiagnostics();
            }

            if (Time.realtimeSinceStartup - m_StartedAt > timeoutSeconds)
            {
                Debug.LogError($"InferenceProbe: за {timeoutSeconds:F0} с собрано только " +
                               $"{m_Rewards.Count} из {episodes} эпизодов — записываю частичный результат.");
                Finish(timedOut: true);
            }
        }

        void LogDiagnostics()
        {
            var agents = FindObjectsByType<AgentBase>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            string first = "нет агентов";
            if (agents.Length > 0)
            {
                var a = agents[0];
                first = $"{a.name}: StepCount={a.StepCount}, CompletedEpisodes={a.CompletedEpisodes}, " +
                        $"MaxStep={a.MaxStep}, enabled={a.isActiveAndEnabled}, result={a.LastEpisodeResult}";
            }

            Debug.Log($"InferenceProbe[диагностика]: агентов={agents.Length}, " +
                      $"Academy.IsInitialized={Academy.IsInitialized}, " +
                      $"AcademyStepCount={(Academy.IsInitialized ? Academy.Instance.TotalStepCount : -1)}, " +
                      $"собрано эпизодов={m_Rewards.Count}, кадров={Time.frameCount}\n  {first}");
        }

        void OnEpisodeFinished(AgentBase agent, float reward, string result)
        {
            if (m_Finished || m_Rewards.Count >= episodes) return;

            m_Rewards.Add(reward);
            m_Outcomes.TryGetValue(result, out int count);
            m_Outcomes[result] = count + 1;

            if (m_Rewards.Count >= episodes) Finish(timedOut: false);
        }

        void Finish(bool timedOut)
        {
            m_Finished = true;

            float sum = 0f, min = float.MaxValue, max = float.MinValue;
            foreach (float r in m_Rewards)
            {
                sum += r;
                if (r < min) min = r;
                if (r > max) max = r;
            }

            int n = m_Rewards.Count;
            float mean = n > 0 ? sum / n : 0f;

            var json = new StringBuilder();
            json.Append("{\n");
            json.AppendFormat(CultureInfo.InvariantCulture, "  \"episodes\": {0},\n", n);
            json.AppendFormat(CultureInfo.InvariantCulture, "  \"requested\": {0},\n", episodes);
            json.AppendFormat(CultureInfo.InvariantCulture, "  \"timed_out\": {0},\n", timedOut ? "true" : "false");
            json.AppendFormat(CultureInfo.InvariantCulture, "  \"mean_reward\": {0:R},\n", mean);
            json.AppendFormat(CultureInfo.InvariantCulture, "  \"min_reward\": {0:R},\n", n > 0 ? min : 0f);
            json.AppendFormat(CultureInfo.InvariantCulture, "  \"max_reward\": {0:R},\n", n > 0 ? max : 0f);

            json.Append("  \"rewards\": [");
            for (int i = 0; i < n; i++)
            {
                if (i > 0) json.Append(", ");
                json.AppendFormat(CultureInfo.InvariantCulture, "{0:R}", m_Rewards[i]);
            }
            json.Append("],\n");

            json.Append("  \"outcomes\": {");
            bool first = true;
            foreach (var pair in m_Outcomes)
            {
                if (!first) json.Append(", ");
                json.AppendFormat(CultureInfo.InvariantCulture, "\"{0}\": {1}", pair.Key, pair.Value);
                first = false;
            }
            json.Append("}\n}\n");

            try
            {
                string full = Path.GetFullPath(resultPath);
                Directory.CreateDirectory(Path.GetDirectoryName(full));
                File.WriteAllText(full, json.ToString());
                Debug.Log($"InferenceProbe: {n} эпизодов, средняя награда {mean:F4}; записано в {full}");
            }
            catch (IOException exc)
            {
                Debug.LogError($"InferenceProbe: не удалось записать результат: {exc.Message}");
            }

            Application.Quit(timedOut ? 1 : 0);
        }

        void ParseCommandLine()
        {
            string[] args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                switch (args[i])
                {
                    case "-inferenceEpisodes":
                        if (int.TryParse(args[i + 1], out int parsed) && parsed > 0)
                        {
                            episodes = parsed;
                            m_Enabled = true;
                        }
                        break;
                    case "-inferenceResult":
                        resultPath = args[i + 1];
                        break;
                    case "-inferenceTimeout":
                        if (float.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float seconds))
                            timeoutSeconds = seconds;
                        break;
                }
            }
        }
    }
}
