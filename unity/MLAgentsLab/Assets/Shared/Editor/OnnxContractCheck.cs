using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace LabRL.EditorTools
{
    /// <summary>
    /// Проверка контракта ONNX **на стороне Unity** (раздел 10 инструкции).
    ///
    /// Python-верификатор (<c>labrl.export.onnx_verify</c>) проверяет тот же
    /// контракт по файлу. Эта проверка отвечает на другой вопрос: принял ли
    /// модель импортёр Unity и видит ли Inference Engine те же имена тензоров.
    /// Расхождение возможно: ONNX может быть валиден, но не пройти импорт.
    ///
    /// Имена тензоров продублированы здесь строковыми константами намеренно:
    /// <c>Unity.MLAgents.Inference.TensorNames</c> объявлен <c>internal</c>
    /// и недоступен из кода проекта. Значения сверены с
    /// <c>com.unity.ml-agents@4.0.3/Runtime/Inference/TensorNames.cs</c>
    /// и зафиксированы в docs/04_ONNX_CONTRACT.md.
    ///
    /// Меню: Tools → RL → Check ONNX Contract (выделите .onnx в Project).
    /// </summary>
    public static class OnnxContractCheck
    {
        const string ActionMasks = "action_masks";
        const string RecurrentIn = "recurrent_in";
        const string ObsPrefix = "obs_";

        const string VersionNumber = "version_number";
        const string MemorySize = "memory_size";
        const string ContinuousActions = "continuous_actions";
        const string DiscreteActions = "discrete_actions";
        const string ContinuousActionOutputShape = "continuous_action_output_shape";
        const string DiscreteActionOutputShape = "discrete_action_output_shape";
        const string DeterministicContinuousActions = "deterministic_continuous_actions";
        const string DeterministicDiscreteActions = "deterministic_discrete_actions";
        const string RecurrentOut = "recurrent_out";

        [MenuItem("Tools/RL/Check ONNX Contract")]
        public static void CheckSelected()
        {
            var assets = Selection.GetFiltered<ModelAsset>(SelectionMode.Assets);
            if (assets.Length == 0)
            {
                Debug.LogWarning("OnnxContractCheck: выделите в окне Project хотя бы одну модель (.onnx).");
                return;
            }

            foreach (var asset in assets)
            {
                var problems = Check(asset, out string summary);
                if (problems.Count == 0)
                    Debug.Log($"OnnxContractCheck: {asset.name} — контракт соблюдён.\n{summary}", asset);
                else
                    Debug.LogError($"OnnxContractCheck: {asset.name} — нарушений: {problems.Count}\n" +
                                   string.Join("\n", problems) + "\n" + summary, asset);
            }
        }

        /// <summary>
        /// Сверяет имена входов и выходов модели с контрактом ML-Agents.
        /// </summary>
        /// <param name="asset">Импортированная модель.</param>
        /// <param name="summary">Человекочитаемая сводка: что за входы и выходы найдены.</param>
        /// <returns>Список нарушений; пустой список означает, что контракт соблюдён.</returns>
        public static List<string> Check(ModelAsset asset, out string summary)
        {
            var problems = new List<string>();
            var model = ModelLoader.Load(asset);

            var inputs = model.inputs.Select(i => i.name).ToList();
            var outputs = model.outputs.Select(o => o.name).ToList();
            summary = $"  входы:  {string.Join(", ", inputs)}\n  выходы: {string.Join(", ", outputs)}";

            int numObs = inputs.Count(n => n.StartsWith(ObsPrefix));
            if (numObs == 0)
                problems.Add($"нет ни одного входа {ObsPrefix}N — Unity не сможет подать наблюдения");

            for (int i = 0; i < numObs; i++)
            {
                string expected = ObsPrefix + i;
                if (!inputs.Contains(expected))
                    problems.Add($"наблюдения пронумерованы с пропуском: нет входа {expected}");
            }

            foreach (string required in new[] { ActionMasks, RecurrentIn })
            {
                if (!inputs.Contains(required))
                    problems.Add($"отсутствует обязательный вход {required}");
            }

            var unknownInputs = inputs.Where(n => !n.StartsWith(ObsPrefix) && n != ActionMasks && n != RecurrentIn);
            foreach (string name in unknownInputs)
                problems.Add($"вход {name} не предусмотрен контрактом");

            foreach (string required in new[] { VersionNumber, MemorySize })
            {
                if (!outputs.Contains(required))
                    problems.Add($"отсутствует обязательный выход {required}");
            }

            bool hasContinuous = outputs.Contains(ContinuousActions);
            bool hasDiscrete = outputs.Contains(DiscreteActions);
            if (!hasContinuous && !hasDiscrete)
                problems.Add($"модель не выдаёт ни {ContinuousActions}, ни {DiscreteActions}");

            if (hasContinuous)
            {
                RequireOutput(outputs, ContinuousActionOutputShape, problems);
                RequireOutput(outputs, DeterministicContinuousActions, problems);
            }

            if (hasDiscrete)
            {
                RequireOutput(outputs, DiscreteActionOutputShape, problems);
                RequireOutput(outputs, DeterministicDiscreteActions, problems);
            }

            var known = new HashSet<string>
            {
                VersionNumber, MemorySize,
                ContinuousActions, ContinuousActionOutputShape, DeterministicContinuousActions,
                DiscreteActions, DiscreteActionOutputShape, DeterministicDiscreteActions,
                RecurrentOut,
            };
            foreach (string name in outputs.Where(n => !known.Contains(n)))
                problems.Add($"выход {name} не предусмотрен контрактом");

            return problems;
        }

        static void RequireOutput(List<string> outputs, string name, List<string> problems)
        {
            if (!outputs.Contains(name))
                problems.Add($"отсутствует обязательный выход {name}");
        }
    }
}
