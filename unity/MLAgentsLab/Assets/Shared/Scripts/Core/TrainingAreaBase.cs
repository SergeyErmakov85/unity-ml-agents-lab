using Unity.MLAgents;
using UnityEngine;

namespace LabRL.Core
{
    /// <summary>
    /// База для тренировочной арены (требования 7.2, 7.3).
    ///
    /// Арена — самодостаточная копия среды. K арен размещаются по сетке с шагом,
    /// исключающим взаимное влияние, и трактуются Python-обёрткой как K
    /// параллельных сред (требование 8.2).
    ///
    /// Что делает база:
    /// * принимает <c>seed</c> и <c>difficulty</c> из Python через
    ///   <c>EnvironmentParametersChannel</c> (требование 7.3);
    /// * даёт каждой арене **свой** детерминированный поток случайных чисел,
    ///   чтобы K арен не проходили один и тот же эпизод синхронно;
    /// * фиксирует индекс арены — он же смещение сида.
    ///
    /// Важно про <see cref="Rng"/>: используется локальный <c>System.Random</c>,
    /// а не <c>UnityEngine.Random</c>. Глобальный генератор Unity один на процесс,
    /// и при K аренах порядок их обращений к нему зависит от порядка вызова
    /// <c>OnEpisodeBegin</c> — то есть воспроизводимость теряется. Локальный
    /// генератор на арену этой проблемы не имеет.
    /// </summary>
    public abstract class TrainingAreaBase : MonoBehaviour
    {
        /// <summary>Имя параметра сида в EnvironmentParametersChannel.</summary>
        public const string SeedParameter = "seed";

        /// <summary>Имя параметра сложности в EnvironmentParametersChannel.</summary>
        public const string DifficultyParameter = "difficulty";

        [Tooltip("Индекс арены. Задаётся сценой при инстанцировании; смещает сид.")]
        public int areaIndex;

        [Tooltip("Сид, используемый когда Python не передал параметр seed (ручной запуск в редакторе).")]
        public int fallbackSeed = 0;

        [Tooltip("Сложность по умолчанию, если Python не передал параметр difficulty.")]
        public float defaultDifficulty = 1f;

        /// <summary>Сид этой арены: базовый сид из Python плюс индекс арены.</summary>
        public int AreaSeed { get; private set; }

        /// <summary>Текущая сложность среды, полученная из Python.</summary>
        public float Difficulty { get; private set; }

        System.Random m_Rng;

        /// <summary>
        /// Детерминированный генератор арены. Единственный источник случайности среды.
        ///
        /// Инициализируется лениво: Unity не гарантирует порядок <c>Awake</c> между
        /// ареной и агентом, а <c>Agent.OnEpisodeBegin()</c> может запросить
        /// случайную клетку раньше, чем у арены выполнится <c>Awake</c>. Ленивый
        /// доступ снимает эту гонку, не завися от порядка выполнения.
        /// </summary>
        protected System.Random Rng
        {
            get
            {
                if (m_Rng == null) InitializeArea();
                return m_Rng;
            }
        }

        protected virtual void Awake()
        {
            InitializeArea();
        }

        /// <summary>
        /// Читает параметры из Python и создаёт генератор арены.
        /// Вызывается в <c>Awake</c>; повторный вызов пересоздаёт генератор
        /// (нужно, если сложность меняется по ходу curriculum learning).
        /// </summary>
        public void InitializeArea()
        {
            var parameters = Academy.Instance.EnvironmentParameters;
            int baseSeed = Mathf.RoundToInt(parameters.GetWithDefault(SeedParameter, fallbackSeed));
            Difficulty = parameters.GetWithDefault(DifficultyParameter, defaultDifficulty);

            AreaSeed = unchecked(baseSeed * 7919 + areaIndex);
            m_Rng = new System.Random(AreaSeed);

            OnAreaInitialized();
        }

        /// <summary>
        /// Точка расширения: вызывается после того, как сид и сложность получены,
        /// но до первого <c>OnEpisodeBegin</c> агента.
        /// </summary>
        protected virtual void OnAreaInitialized() { }

        /// <summary>Случайное целое в [minInclusive, maxExclusive) из генератора арены.</summary>
        protected int NextInt(int minInclusive, int maxExclusive) => Rng.Next(minInclusive, maxExclusive);

        /// <summary>Случайное число в [0, 1) из генератора арены.</summary>
        protected float NextFloat() => (float)Rng.NextDouble();

        /// <summary>Случайное число в [min, max) из генератора арены.</summary>
        protected float NextFloat(float min, float max) => min + (max - min) * NextFloat();
    }
}
