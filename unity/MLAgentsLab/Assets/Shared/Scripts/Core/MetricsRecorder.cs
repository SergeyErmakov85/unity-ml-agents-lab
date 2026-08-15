using Unity.MLAgents;

namespace LabRL.Core
{
    /// <summary>
    /// Публикация метрик со стороны среды (требование 7.4).
    ///
    /// Значения уходят в <c>Academy.Instance.StatsRecorder</c>, оттуда — через
    /// <c>StatsSideChannel</c> в Python, и логируются в неймспейсе <c>Env/</c>
    /// (схема 11.4). Имена метрик задаются без префикса: префикс добавляет
    /// Python-сторона, чтобы среда не знала о схеме TensorBoard.
    ///
    /// Обёртка нужна ровно затем, чтобы среда не обращалась к Academy напрямую:
    /// в headless-сборке без коммуникатора StatsRecorder существует, но её вызов
    /// в каждом кадре из десятков арен — заметная нагрузка, поэтому здесь
    /// собраны все точки записи.
    /// </summary>
    public static class MetricsRecorder
    {
        /// <summary>Среднее за период сводки (частота задаётся тренером).</summary>
        public static void Average(string key, float value)
        {
            Academy.Instance.StatsRecorder.Add(key, value, StatAggregationMethod.Average);
        }

        /// <summary>Сумма за период сводки — для счётчиков событий.</summary>
        public static void Sum(string key, float value)
        {
            Academy.Instance.StatsRecorder.Add(key, value, StatAggregationMethod.Sum);
        }

        /// <summary>Последнее значение — для величин состояния (например, текущей сложности).</summary>
        public static void MostRecent(string key, float value)
        {
            Academy.Instance.StatsRecorder.Add(key, value, StatAggregationMethod.MostRecent);
        }

        /// <summary>Гистограмма — для распределений (время до цели, длина эпизода).</summary>
        public static void Histogram(string key, float value)
        {
            Academy.Instance.StatsRecorder.Add(key, value, StatAggregationMethod.Histogram);
        }

        /// <summary>
        /// Доля успешных эпизодов: 1 — успех, 0 — нет. Усредняется за период сводки,
        /// поэтому в TensorBoard попадает именно доля, а не счётчик.
        /// </summary>
        public static void Success(bool succeeded, string key = "SuccessRate")
        {
            Average(key, succeeded ? 1f : 0f);
        }
    }
}
