using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;

namespace LabRL.Core
{
    /// <summary>
    /// База агента лаборатории (требование 7.2).
    ///
    /// Даёт три вещи, одинаковые для всех сред:
    /// 1. проверку главного инварианта проекта — Behavior Name обязан совпадать
    ///    с идентификатором среды <c>E##_&lt;Name&gt;</c> (правило 5.3); это
    ///    единственная точка связи Python ↔ Unity, и её расхождение проявляется
    ///    как «Python не видит агентов», без внятного сообщения об ошибке;
    /// 2. учёт результата эпизода и публикацию <c>Env/</c>-метрик (требование 7.4);
    /// 3. ссылку на арену, которой принадлежит агент.
    ///
    /// Наследник обязан реализовать <see cref="EnvId"/> и обычные методы
    /// <c>Agent</c>: <c>OnEpisodeBegin</c>, <c>CollectObservations</c>,
    /// <c>OnActionReceived</c>, <c>Heuristic</c>.
    /// </summary>
    public abstract class AgentBase : Agent
    {
        /// <summary>Идентификатор среды, например <c>E03_RollerBall</c>. Он же Behavior Name.</summary>
        public abstract string EnvId { get; }

        /// <summary>Арена, которой принадлежит агент. Может быть null у сред с единственной ареной.</summary>
        public TrainingAreaBase Area { get; private set; }

        /// <summary>Чем закончился последний завершённый эпизод — для отладочного HUD.</summary>
        public string LastEpisodeResult { get; protected set; } = "Running";

        BehaviorParameters m_Behavior;
        int m_EpisodeStartStep;

        public override void Initialize()
        {
            base.Initialize();
            Area = GetComponentInParent<TrainingAreaBase>();
            m_Behavior = GetComponent<BehaviorParameters>();
            AssertBehaviorName();
        }

        /// <summary>
        /// Сверяет Behavior Name с <see cref="EnvId"/>. Расхождение — ошибка конфигурации
        /// сцены, а не повод «продолжить как-нибудь»: обучение молча не запустится.
        /// </summary>
        void AssertBehaviorName()
        {
            if (m_Behavior == null)
            {
                Debug.LogError($"{name}: на агенте нет компонента BehaviorParameters.", this);
                return;
            }

            if (m_Behavior.BehaviorName != EnvId)
            {
                Debug.LogError(
                    $"{name}: Behavior Name = \"{m_Behavior.BehaviorName}\", ожидался \"{EnvId}\". " +
                    "Python подключается к среде именно по этому имени (правило 5.3) — исправьте " +
                    "Behavior Parameters или пересоберите сцену её Setup-скриптом.", this);
            }
        }

        /// <summary>
        /// Завершает эпизод, публикуя стандартные метрики среды в неймспейс <c>Env/</c>.
        /// </summary>
        /// <param name="result">Метка исхода: <c>Goal</c>, <c>Trap</c>, <c>Timeout</c> и т. п.</param>
        /// <param name="success">Считается ли исход успешным (идёт в <c>Env/SuccessRate</c>).</param>
        protected void EndEpisodeWithResult(string result, bool success)
        {
            LastEpisodeResult = result;
            MetricsRecorder.Success(success);
            MetricsRecorder.Histogram("EpisodeSteps", StepCount - m_EpisodeStartStep);
            EndEpisode();
        }

        /// <summary>
        /// Наследник обязан вызвать <c>base.OnEpisodeBegin()</c>, если переопределяет метод:
        /// здесь фиксируется точка отсчёта длины эпизода.
        /// </summary>
        public override void OnEpisodeBegin()
        {
            m_EpisodeStartStep = StepCount;
            LastEpisodeResult = "Running";
        }
    }
}
