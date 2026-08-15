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
        /// <summary>
        /// Событие завершения эпизода: агент, суммарная награда, метка исхода.
        ///
        /// Нужно для проверки инференса в Unity (требование 10.6): внешний
        /// наблюдатель должен получать награду **завершённого** эпизода, а
        /// <c>GetCumulativeReward()</c> обнуляется в момент начала следующего,
        /// и опросом его надёжно не поймать.
        ///
        /// Событие статическое: наблюдатель один на сцену, а агентов — по числу
        /// арен, и подписываться на каждого пришлось бы через поиск объектов.
        /// Обработчики обязаны отписываться в <c>OnDisable</c>, иначе при
        /// перезагрузке домена в редакторе накопятся мёртвые подписки.
        /// </summary>
        public static event System.Action<AgentBase, float, string> EpisodeFinished;

        /// <summary>Идентификатор среды, например <c>E03_RollerBall</c>. Он же Behavior Name.</summary>
        public abstract string EnvId { get; }

        /// <summary>Арена, которой принадлежит агент. Может быть null у сред с единственной ареной.</summary>
        public TrainingAreaBase Area { get; private set; }

        /// <summary>Чем закончился последний завершённый эпизод — для отладочного HUD.</summary>
        public string LastEpisodeResult { get; protected set; } = RunningResult;

        /// <summary>Метка незавершённого эпизода. Ею же различается обрыв по времени.</summary>
        public const string RunningResult = "Running";

        BehaviorParameters m_Behavior;
        int m_EpisodeStartStep;
        int m_StepsAtEpisodeEnd;
        float m_RewardAtEpisodeEnd;
        bool m_EpisodeStarted;

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
            // Событие поднимается до EndEpisode(): после него награда эпизода
            // уже обнулена и вернуть её нельзя.
            EpisodeFinished?.Invoke(this, GetCumulativeReward(), result);
            EndEpisode();
        }

        /// <summary>
        /// Наследник обязан вызвать <c>base.OnEpisodeBegin()</c>, если переопределяет метод:
        /// здесь фиксируется точка отсчёта длины эпизода и закрывается предыдущий
        /// эпизод, если тот оборвался по <c>MaxStep</c>.
        ///
        /// Почему обрыв по времени обрабатывается **здесь**, а не в
        /// <c>OnActionReceived</c>. ML-Agents завершает эпизод по <c>MaxStep</c>
        /// в фазе <c>AgentPreStep</c>, то есть **раньше**, чем вызывается
        /// <c>OnActionReceived</c> этого шага. Проверка вида
        /// <c>if (StepCount >= MaxStep)</c> внутри <c>OnActionReceived</c> —
        /// мёртвый код: она не срабатывает никогда, и обрывы по времени тихо
        /// исчезают из статистики, завышая долю успехов. Единственная точка,
        /// где факт обрыва ещё виден, — начало следующего эпизода: у предыдущего
        /// так и остался результат <c>Running</c>.
        /// </summary>
        public override void OnEpisodeBegin()
        {
            if (m_EpisodeStarted && LastEpisodeResult == RunningResult)
            {
                LastEpisodeResult = "Timeout";
                MetricsRecorder.Success(false);
                MetricsRecorder.Histogram("EpisodeSteps", m_StepsAtEpisodeEnd - m_EpisodeStartStep);
                EpisodeFinished?.Invoke(this, m_RewardAtEpisodeEnd, "Timeout");
            }

            m_EpisodeStarted = true;
            m_EpisodeStartStep = StepCount;
            LastEpisodeResult = RunningResult;
        }

        /// <summary>
        /// Запоминает награду и число шагов на случай, если эпизод оборвётся
        /// по <c>MaxStep</c>: к моменту следующего <c>OnEpisodeBegin</c>
        /// <c>GetCumulativeReward()</c> и <c>StepCount</c> уже обнулены.
        /// Вызывается в конце <c>OnActionReceived</c> наследника через
        /// <see cref="TrackStep"/>.
        /// </summary>
        protected void TrackStep()
        {
            m_RewardAtEpisodeEnd = GetCumulativeReward();
            m_StepsAtEpisodeEnd = StepCount;
        }
    }
}
