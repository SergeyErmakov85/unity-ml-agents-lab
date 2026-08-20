using System.Collections.Generic;
using LabRL.Core;
using Unity.MLAgents;
using UnityEngine;

/// <summary>
/// Арена среды `E08_SoccerArena` (TS-008) — поле 24 × 16, двое ворот, мяч
/// и две команды по два игрока.
///
/// Отвечает за то, что не принадлежит отдельному игроку: расстановку,
/// обнаружение гола, командную награду и синхронное завершение эпизода
/// у всех четверых.
///
/// **Почему завершение эпизода живёт здесь, а не в агенте.** Гол — событие
/// матча, а не событие игрока. Если каждый агент завершал бы эпизод сам,
/// четыре завершения разъехались бы по кадрам, и Python получил бы
/// терминальные шаги четырёх агентов в разные моменты — то есть переходы
/// одного и того же матча попали бы в разные эпизоды.
///
/// **Командная награда идёт через <see cref="SimpleMultiAgentGroup"/>.**
/// Это не косметика: MA-POCA обучается именно на групповой награде, и её
/// подмена суммой индивидуальных лишает алгоритм того, ради чего он существует —
/// задачи распределения заслуги внутри команды (урок 3.2).
/// </summary>
public class SoccerArea : TrainingAreaBase
{
    /// <summary>Сторона поля. Совпадает с <c>TeamId</c> в <c>BehaviorParameters</c>.</summary>
    public enum Side
    {
        /// <summary>Защищает ворота при x = −halfLength, атакует правые. TeamId = 0.</summary>
        West = 0,

        /// <summary>Защищает ворота при x = +halfLength, атакует левые. TeamId = 1.</summary>
        East = 1,
    }

    [Header("Ссылки")]
    public Rigidbody ball;
    public Transform ballSpawn;

    [Tooltip("Игроки команды West (TeamId = 0).")]
    public List<SoccerPlayerAgent> westPlayers = new List<SoccerPlayerAgent>(2);

    [Tooltip("Игроки команды East (TeamId = 1).")]
    public List<SoccerPlayerAgent> eastPlayers = new List<SoccerPlayerAgent>(2);

    [Header("Геометрия")]
    [Tooltip("Половина длины поля вдоль X. Поле 24 в длину -> 12.")]
    public float halfLength = 12f;

    [Tooltip("Половина ширины поля вдоль Z. Поле 16 в ширину -> 8.")]
    public float halfWidth = 8f;

    [Tooltip("Высота, на которой стоят игроки и мяч.")]
    public float spawnHeight = 0.5f;

    [Header("Спавн")]
    [Tooltip("Максимальное случайное смещение мяча от центра по каждой оси.")]
    public float ballSpawnJitter = 1.5f;

    [Tooltip("Минимальная дистанция между точками спавна игроков и мячом.")]
    public float minSeparation = 2f;

    [Header("Награда")]
    [Tooltip("Награда группе за гол в чужие ворота при t = 0. К концу эпизода " +
             "убывает до (1 − goalTimeDiscount).")]
    public float goalReward = 1f;

    [Tooltip("Насколько дешевеет гол к концу эпизода: 0.5 -> поздний гол стоит 0.5.")]
    public float goalTimeDiscount = 0.5f;

    [Tooltip("Награда группе, в чьи ворота забили. Отрицательна.")]
    public float concedeReward = -1f;

    [Tooltip("Вес потенциального формирования награды по положению мяча. " +
             "0 — выключено (чистая редкая награда).")]
    public float shapingWeight = 0.3f;

    [Tooltip("γ формирования награды. ОБЯЗАНА совпадать с gamma алгоритма обучения: " +
             "теорема Ына о неизменности оптимальной политики доказана именно для этого γ.")]
    public float shapingGamma = 0.99f;

    /// <summary>Группа команды West. Через неё начисляется командная награда.</summary>
    public SimpleMultiAgentGroup WestGroup { get; private set; }

    /// <summary>Группа команды East.</summary>
    public SimpleMultiAgentGroup EastGroup { get; private set; }

    readonly List<Vector3> m_Occupied = new List<Vector3>(8);
    bool m_EpisodeResolved;
    Side? m_LastTouch;
    // Потенциал предыдущего шага, отдельно для каждой стороны: он у них
    // противоположен по знаку (мяч у чужих ворот хорош ровно для одной команды).
    readonly float[] m_PreviousPotential = new float[2];

    void Start()
    {
        // Группы создаются в Start, а не в Awake: Agent.Initialize() выполняется
        // в Awake-фазе, и регистрация агента в группе до его инициализации
        // не гарантирована по порядку.
        WestGroup = new SimpleMultiAgentGroup();
        EastGroup = new SimpleMultiAgentGroup();

        foreach (var player in westPlayers) WestGroup.RegisterAgent(player);
        foreach (var player in eastPlayers) EastGroup.RegisterAgent(player);

        ResetMatch();
    }

    void OnDestroy()
    {
        WestGroup?.Dispose();
        EastGroup?.Dispose();
    }

    /// <summary>Группа стороны.</summary>
    public SimpleMultiAgentGroup GroupOf(Side side) => side == Side.West ? WestGroup : EastGroup;

    /// <summary>Мяч в мировых координатах арены (локальных относительно неё).</summary>
    public Vector3 BallPosition => ball.transform.localPosition;

    /// <summary>Скорость мяча в системе координат арены.</summary>
    public Vector3 BallVelocity => transform.InverseTransformDirection(ball.linearVelocity);

    /// <summary>
    /// Знак оси X для стороны: команда East смотрит на мир зеркально,
    /// чтобы «вперёд, к чужим воротам» для обеих команд было <c>+x</c> (ТЗ §4).
    /// </summary>
    public static float MirrorSign(Side side) => side == Side.West ? 1f : -1f;

    /// <summary>
    /// Потенциал состояния для стороны: положение мяча вдоль поля в командной
    /// системе координат, Φ ∈ [−1, 1]. Мяч у чужих ворот даёт +1.
    /// </summary>
    public float Potential(Side side) =>
        Mathf.Clamp(MirrorSign(side) * BallPosition.x / halfLength, -1f, 1f);

    /// <summary>
    /// Начисляет группе формирующую награду за продвижение мяча.
    ///
    /// **Зачем она здесь.** «Забил / не забил» на горизонте 600 решений —
    /// сигнал, которого случайная политика не получит никогда: измерено на
    /// живом билде, 1400 шагов случайной игры дали 8 завершений и ноль голов.
    /// Без плотной награды обучение стартует с нулевого градиента.
    ///
    /// **Почему именно в потенциальной форме** F(s, s′) = γ·Φ(s′) − Φ(s).
    /// Наивная награда «+ за приближение мяча к воротам» создаёт reward
    /// hacking: откатить мяч назад бесплатно, подтолкнуть снова — выгодно.
    /// Потенциальная форма телескопируется и доказуемо не меняет множество
    /// оптимальных политик (Ng, Harada, Russell, 1999) — то же обоснование,
    /// что в `E06_Hunter3D`.
    ///
    /// **Награда командная**, а не личная: мяч продвигают оба игрока, и кто
    /// именно внёс вклад — вопрос контрфактического базлайна MA-POCA,
    /// а не функции награды.
    ///
    /// Вызывается ровно один раз за шаг MDP — «докладчиком» команды
    /// (<see cref="SoccerPlayerAgent.shapingReporter"/>): начислить её
    /// от каждого игрока значило бы умножить на размер команды.
    /// </summary>
    public void ApplyBallShaping(Side side)
    {
        if (shapingWeight <= 0f || m_EpisodeResolved) return;

        int index = (int)side;
        float next = Potential(side);
        GroupOf(side).AddGroupReward(shapingWeight * (shapingGamma * next - m_PreviousPotential[index]));
        m_PreviousPotential[index] = next;
    }

    /// <summary>
    /// Запоминает, игрок какой стороны последним коснулся мяча.
    /// Нужно только для отличия автогола от обычного — награды у них
    /// одинаковы (ТЗ §6), различается метрика.
    /// </summary>
    public void NoteBallTouch(Side side)
    {
        m_LastTouch = side;
        MetricsRecorder.Sum("BallTouches", 1f);
    }

    /// <summary>
    /// Гол: мяч пересёк створ. Вызывается триггером ворот
    /// (<see cref="SoccerGoal"/>).
    /// </summary>
    /// <param name="scoredOn">Сторона, в чьи ворота залетел мяч.</param>
    public void RegisterGoal(Side scoredOn)
    {
        if (!m_EpisodeResolved && m_LastTouch == scoredOn)
            MetricsRecorder.Sum("OwnGoals", 1f);

        // Триггер может сработать несколько раз за кадр (мяч касается двух
        // коллайдеров створа), а также после того, как эпизод уже завершён,
        // но объекты ещё не расставлены. Флаг закрывает оба случая.
        if (m_EpisodeResolved) return;
        m_EpisodeResolved = true;

        Side scorer = scoredOn == Side.West ? Side.East : Side.West;

        // Потенциал терминального состояния обязан быть нулём — иначе сумма
        // формирующих наград за эпизод перестаёт телескопироваться и смещает
        // итоговую награду. Здесь это доначисление −Φ(s) обеим сторонам.
        if (shapingWeight > 0f)
        {
            for (int i = 0; i < 2; i++)
            {
                var side = (Side)i;
                GroupOf(side).AddGroupReward(shapingWeight * (0f - m_PreviousPotential[i]));
                m_PreviousPotential[i] = 0f;
            }
        }

        // Доля израсходованного времени берётся у первого игрока: MaxStep
        // и StepCount у всех четверых одинаковы, потому что эпизод общий.
        float progress = westPlayers.Count > 0 && westPlayers[0].MaxStep > 0
            ? Mathf.Clamp01((float)westPlayers[0].StepCount / westPlayers[0].MaxStep)
            : 0f;
        float reward = goalReward * (1f - goalTimeDiscount * progress);

        GroupOf(scorer).AddGroupReward(reward);
        GroupOf(scoredOn).AddGroupReward(concedeReward);

        MetricsRecorder.Sum(scorer == Side.West ? "GoalsWest" : "GoalsEast", 1f);
        MetricsRecorder.Success(true);

        // Каждый игрок помечает свой исход до завершения группы: после
        // EndGroupEpisode накопленная награда обнуляется, и наблюдатель
        // инференса (требование 10.6) её уже не увидит.
        foreach (var player in westPlayers) player.NoteMatchResult(scorer == Side.West ? "Goal" : "Conceded");
        foreach (var player in eastPlayers) player.NoteMatchResult(scorer == Side.East ? "Goal" : "Conceded");

        WestGroup.EndGroupEpisode();
        EastGroup.EndGroupEpisode();
        ResetMatch();
    }

    /// <summary>
    /// Обрыв матча по времени. Вызывается игроком, у которого истёк
    /// <c>MaxStep</c>: он один и тот же у всех четверых.
    ///
    /// Именно <c>GroupEpisodeInterrupted</c>, а не <c>EndGroupEpisode</c>:
    /// обрыв по времени — не терминальное состояние, ценность будущего
    /// не равна нулю, и бутстрэппинг обязан выполняться (ТЗ §6).
    /// </summary>
    public void RegisterTimeout()
    {
        if (m_EpisodeResolved) return;
        m_EpisodeResolved = true;

        WestGroup.GroupEpisodeInterrupted();
        EastGroup.GroupEpisodeInterrupted();
        ResetMatch();
    }

    /// <summary>Расставляет мяч и игроков; снимает флаг завершённого матча.</summary>
    public void ResetMatch()
    {
        ball.linearVelocity = Vector3.zero;
        ball.angularVelocity = Vector3.zero;

        Vector3 ballPoint = ballSpawn.localPosition + new Vector3(
            NextFloat(-ballSpawnJitter, ballSpawnJitter), 0f,
            NextFloat(-ballSpawnJitter, ballSpawnJitter));
        ballPoint.y = spawnHeight;
        ball.transform.localPosition = ballPoint;

        m_Occupied.Clear();
        m_Occupied.Add(ballPoint);

        PlaceTeam(westPlayers, Side.West);
        PlaceTeam(eastPlayers, Side.East);

        m_LastTouch = null;
        m_PreviousPotential[0] = Potential(Side.West);
        m_PreviousPotential[1] = Potential(Side.East);
        m_EpisodeResolved = false;
    }

    /// <summary>
    /// Ставит игроков команды в свою половину: West — при отрицательных X,
    /// East — при положительных. Половины не пересекаются, поэтому стартовая
    /// позиция никогда не даёт одной команде преимущества у чужих ворот.
    /// </summary>
    void PlaceTeam(List<SoccerPlayerAgent> players, Side side)
    {
        float sign = MirrorSign(side);
        // Половина по X: от четверти поля до почти самой линии ворот.
        var halfExtents = new Vector2(0.25f * halfLength, halfWidth - 1.5f);
        Vector3 center = new Vector3(-sign * 0.5f * halfLength, spawnHeight, 0f);

        foreach (var player in players)
        {
            Vector3 point = SpawnService.RandomPointAwayFrom(
                Rng, center, halfExtents, spawnHeight, m_Occupied, minSeparation, out _);
            m_Occupied.Add(point);

            var body = player.GetComponent<Rigidbody>();
            body.linearVelocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
            player.transform.localPosition = point;
            // Стартовый курс — «в сторону чужих ворот» плюс разброс: полностью
            // случайный курс тратит первые решения на разворот и зашумляет
            // и без того редкий сигнал гола.
            float baseYaw = side == Side.West ? 90f : -90f;
            player.transform.localRotation = Quaternion.Euler(0f, baseYaw + NextFloat(-45f, 45f), 0f);
        }
    }
}
