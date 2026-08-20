using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Собирает сцену среды `E08_SoccerArena` по TS-008: поле 24 × 16, двое ворот
/// со створом, мяч и две команды по два игрока, K тренировочных арен из одного
/// префаба.
/// Меню: Tools → RL → Build Soccer Scene.
///
/// Сцена строится кодом целиком: чтобы её изменить, меняется этот скрипт
/// и сцена пересобирается.
/// </summary>
public static class SoccerSetup
{
    /// <summary>Идентификатор среды. Behavior Name обязан совпадать с ним (правило 5.3).</summary>
    private const string EnvId = "E08_SoccerArena";

    private const string Root = "Assets/Envs/E08_SoccerArena";
    private const string ScenePath = Root + "/Scenes/E08_SoccerArena.unity";

    /// <summary>Арен в сцене. 4 арены × 4 игрока = 16 агентов, по 8 на команду.</summary>
    private const int AreaCount = 4;

    private const int PlayersPerTeam = 2;

    private const float HalfLength = 12f;
    private const float HalfWidth = 8f;

    /// <summary>Полуширина створа ворот по Z.</summary>
    private const float GoalHalfWidth = 2.75f;

    /// <summary>Шаг размещения арен по X. Поле длиной 24, шаг 30 даёт зазор 6.</summary>
    private const float AreaSpacing = 30f;

    private const int MaxStep = 3000;
    private const int DecisionPeriod = 5;

    // --- лучевой сенсор ---------------------------------------------------
    /// <summary>Лучей в каждую сторону от центрального: 3 -> всего 7 лучей.</summary>
    private const int RaysPerDirection = 3;

    /// <summary>Половина угла веера: 90° -> обзор 180°.</summary>
    private const float MaxRayDegrees = 90f;

    private const float RayLength = 20f;

    /// <summary>
    /// Порядок тегов для команды West. Позиция в списке — смысл для агента:
    /// 2 — свой, 3 — соперник, 4 — чужие ворота, 5 — свои (ТЗ §4).
    /// </summary>
    private static readonly string[] TagsWest =
        { "ball", "wall", "playerWest", "playerEast", "goalEast", "goalWest" };

    /// <summary>Тот же список, зеркально переставленный для команды East.</summary>
    private static readonly string[] TagsEast =
        { "ball", "wall", "playerEast", "playerWest", "goalWest", "goalEast" };

    private static readonly Color ColField = new Color(0.18f, 0.35f, 0.22f);
    private static readonly Color ColWall = new Color(0.55f, 0.57f, 0.62f);
    private static readonly Color ColBall = new Color(0.95f, 0.93f, 0.88f);
    private static readonly Color ColWest = new Color(0.25f, 0.55f, 0.95f);
    private static readonly Color ColEast = new Color(0.85f, 0.35f, 0.70f);
    private static readonly Color ColGoalWest = new Color(0.20f, 0.40f, 0.75f);
    private static readonly Color ColGoalEast = new Color(0.70f, 0.25f, 0.55f);

    [MenuItem("Tools/RL/Build Soccer Scene")]
    public static void BuildScene()
    {
        EnsureTags();
        EnsureFolder(Root + "/Materials");
        EnsureFolder(Root + "/Scenes");
        EnsureFolder(Root + "/Prefabs");

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Арена как шаблон ----------
        var area = new GameObject("TrainingArea");
        var areaScript = area.AddComponent<SoccerArea>();
        areaScript.halfLength = HalfLength;
        areaScript.halfWidth = HalfWidth;

        var field = GameObject.CreatePrimitive(PrimitiveType.Plane);
        field.name = "Field";
        field.transform.SetParent(area.transform, false);
        // Plane в Unity — 10 × 10 единиц при масштабе 1.
        field.transform.localScale = new Vector3(2f * HalfLength / 10f, 1f, 2f * HalfWidth / 10f);
        field.GetComponent<Renderer>().sharedMaterial = MakeMaterial("FieldMat", ColField);

        // ---------- Борта ----------
        var walls = new GameObject("Walls");
        walls.transform.SetParent(area.transform, false);

        float sideSpan = 2f * HalfLength + 0.5f;
        CreateBox(walls.transform, "Side_North", new Vector3(0f, 1f, HalfWidth + 0.25f),
                  new Vector3(sideSpan, 2f, 0.5f), "WallMat", ColWall, "wall");
        CreateBox(walls.transform, "Side_South", new Vector3(0f, 1f, -HalfWidth - 0.25f),
                  new Vector3(sideSpan, 2f, 0.5f), "WallMat", ColWall, "wall");

        // Задние стены разбиты на два сегмента: между ними — открытый створ,
        // куда встаёт триггер ворот. Сплошная стена отбивала бы мяч, и гол
        // не случился бы никогда.
        float backSegment = HalfWidth - GoalHalfWidth;
        float backOffset = GoalHalfWidth + 0.5f * backSegment;
        foreach (var (name, x) in new[] { ("West", -HalfLength - 0.25f), ("East", HalfLength + 0.25f) })
        {
            CreateBox(walls.transform, $"Back_{name}_North", new Vector3(x, 1f, backOffset),
                      new Vector3(0.5f, 2f, backSegment), "WallMat", ColWall, "wall");
            CreateBox(walls.transform, $"Back_{name}_South", new Vector3(x, 1f, -backOffset),
                      new Vector3(0.5f, 2f, backSegment), "WallMat", ColWall, "wall");
        }

        // ---------- Ворота ----------
        var goals = new GameObject("Goals");
        goals.transform.SetParent(area.transform, false);
        CreateGoal(goals.transform, areaScript, SoccerArea.Side.West,
                   new Vector3(-HalfLength - 0.5f, 1f, 0f), "goalWest", "GoalWestMat", ColGoalWest);
        CreateGoal(goals.transform, areaScript, SoccerArea.Side.East,
                   new Vector3(HalfLength + 0.5f, 1f, 0f), "goalEast", "GoalEastMat", ColGoalEast);

        // ---------- Мяч ----------
        var ballGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        ballGo.name = "Ball";
        ballGo.tag = "ball";
        ballGo.transform.SetParent(area.transform, false);
        ballGo.transform.localPosition = new Vector3(0f, 0.5f, 0f);
        ballGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("BallMat", ColBall);
        var ballBody = ballGo.AddComponent<Rigidbody>();
        ballBody.mass = 0.5f;
        // Мяч должен катиться, но не бесконечно: без демпфирования один удар
        // отправляет его в бортовой пинг-понг на весь эпизод.
        ballBody.linearDamping = 0.5f;
        ballBody.angularDamping = 0.5f;
        ballBody.constraints = RigidbodyConstraints.FreezePositionY;
        areaScript.ball = ballBody;

        var ballSpawn = new GameObject("BallSpawn");
        ballSpawn.transform.SetParent(area.transform, false);
        ballSpawn.transform.localPosition = new Vector3(0f, 0.5f, 0f);
        areaScript.ballSpawn = ballSpawn.transform;

        // ---------- Игроки ----------
        areaScript.westPlayers = new List<SoccerPlayerAgent>(PlayersPerTeam);
        areaScript.eastPlayers = new List<SoccerPlayerAgent>(PlayersPerTeam);

        var westRoot = new GameObject("TeamWest");
        westRoot.transform.SetParent(area.transform, false);
        var eastRoot = new GameObject("TeamEast");
        eastRoot.transform.SetParent(area.transform, false);

        for (int i = 0; i < PlayersPerTeam; i++)
        {
            areaScript.westPlayers.Add(CreatePlayer(
                westRoot.transform, areaScript, SoccerArea.Side.West, i,
                new Vector3(-0.5f * HalfLength, 0.5f, (i == 0 ? 1f : -1f) * 0.4f * HalfWidth),
                90f, "playerWest", "PlayerWestMat", ColWest, TagsWest,
                // С клавиатуры управляется ровно один игрок: одна клавиша,
                // двигающая всех четверых, ручную проверку не даёт.
                keyboard: i == 0));

            areaScript.eastPlayers.Add(CreatePlayer(
                eastRoot.transform, areaScript, SoccerArea.Side.East, i,
                new Vector3(0.5f * HalfLength, 0.5f, (i == 0 ? -1f : 1f) * 0.4f * HalfWidth),
                -90f, "playerEast", "PlayerEastMat", ColEast, TagsEast,
                keyboard: false));
        }

        // ---------- Префаб и K арен ----------
        var areaPrefab = PrefabUtility.SaveAsPrefabAsset(area, Root + "/Prefabs/TrainingArea.prefab");
        Object.DestroyImmediate(area);

        var rootTrainingAreas = new GameObject("TrainingAreas");
        var rootCameras = new GameObject("Cameras");
        var rootLighting = new GameObject("Lighting");
        new GameObject("Environment");
        new GameObject("Managers");
        new GameObject("UI");
        new GameObject("Debug");

        for (int i = 0; i < AreaCount; i++)
        {
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(areaPrefab, rootTrainingAreas.transform);
            instance.name = $"TrainingArea_{i:00}";
            instance.transform.localPosition = new Vector3(i * AreaSpacing, 0f, 0f);
            instance.GetComponent<SoccerArea>().areaIndex = i;  // смещение сида арены

            // Клавиатурой управляется игрок только первой арены: иначе одно
            // нажатие двигало бы по игроку в каждой из K арен.
            if (i > 0)
            {
                foreach (var player in instance.GetComponentsInChildren<SoccerPlayerAgent>())
                    player.keyboardControlled = false;
            }
        }

        // ---------- Камера ----------
        var cam = Camera.main;
        cam.gameObject.name = "MainCamera";
        cam.transform.SetParent(rootCameras.transform, true);
        cam.transform.position = new Vector3(0f, 26f, -22f);
        cam.transform.rotation = Quaternion.Euler(50f, 0f, 0f);
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.10f, 0.11f, 0.14f);

        // ---------- Освещение ----------
        var lightGo = GameObject.Find("Directional Light");
        lightGo.name = "DirectionalLight";
        lightGo.transform.SetParent(rootLighting.transform, true);
        var light = lightGo.GetComponent<Light>();
        light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        light.intensity = 1f;
        light.shadows = LightShadows.Soft;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.25f, 0.25f, 0.31f);

        EditorSceneManager.SaveScene(scene, ScenePath);
        AssetDatabase.SaveAssets();

        int rays = 2 * RaysPerDirection + 1;
        Debug.Log($"Soccer-сцена собрана: {ScenePath}; арен: {AreaCount}, " +
                  $"агентов: {AreaCount * 2 * PlayersPerTeam} (по {AreaCount * PlayersPerTeam} на команду); " +
                  $"лучей: {rays}, тегов: {TagsWest.Length}, размер лучевого наблюдения: {rays * (TagsWest.Length + 2)}");
    }

    private static SoccerPlayerAgent CreatePlayer(
        Transform parent, SoccerArea area, SoccerArea.Side side, int index,
        Vector3 localPosition, float yaw, string tag, string materialName, Color color,
        string[] detectableTags, bool keyboard)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = $"Player_{index}";
        go.tag = tag;
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPosition;
        go.transform.localRotation = Quaternion.Euler(0f, yaw, 0f);
        go.GetComponent<Renderer>().sharedMaterial = MakeMaterial(materialName, color);

        var body = go.AddComponent<Rigidbody>();
        body.mass = 2f;
        // Демпфирование заменяет трение: без него игрок бесконечно скользит
        // по инерции, и управление тягой перестаёт быть управлением.
        body.linearDamping = 1.5f;
        body.angularDamping = 4f;
        // Поворот задаётся действием через transform.Rotate, поэтому вращение
        // от физики заморожено — иначе куб заваливается от толчков о борта.
        body.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ
                         | RigidbodyConstraints.FreezePositionY;

        var agent = go.AddComponent<SoccerPlayerAgent>();
        agent.area = area;
        agent.side = side;
        agent.keyboardControlled = keyboard;
        // «Докладчик» командной формирующей награды — ровно один игрок команды.
        agent.shapingReporter = index == 0;
        agent.MaxStep = MaxStep;
        agent.decisionPeriod = DecisionPeriod;
        // Личный штраф за решение: −0.5 за эпизод из MaxStep / DecisionPeriod решений.
        agent.stepPenalty = -0.5f * DecisionPeriod / MaxStep;

        // Лучи: «зрение» игрока. Имя сенсора влияет на порядок наблюдений —
        // ML-Agents сортирует сенсоры по имени (Agent.InitializeSensors),
        // поэтому "RayPerceptionSensor" идёт перед "VectorSensor_size8"
        // и становится obs_0.
        var rays = go.AddComponent<RayPerceptionSensorComponent3D>();
        rays.SensorName = "RayPerceptionSensor";
        // Порядок тегов зеркальный у двух команд — в этом весь приём (ТЗ §4).
        rays.DetectableTags = new List<string>(detectableTags);
        rays.RaysPerDirection = RaysPerDirection;
        rays.MaxRayDegrees = MaxRayDegrees;
        rays.RayLength = RayLength;
        rays.SphereCastRadius = 0.4f;
        rays.ObservationStacks = 1;
        rays.StartVerticalOffset = 0.3f;
        rays.EndVerticalOffset = 0.3f;

        var behavior = go.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = EnvId;
        // TeamId разводит команды в два поведения на стороне Python:
        // E08_SoccerArena?team=0 и ?team=1 (BehaviorParameters.FullyQualifiedBehaviorName).
        // Это и есть механизм self-play.
        behavior.TeamId = (int)side;
        behavior.BrainParameters.VectorObservationSize = 8;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(3, 3, 3);
        behavior.BehaviorType = BehaviorType.Default;

        var requester = go.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = DecisionPeriod;
        // Действие НЕ повторяется между решениями: решение — это шаг MDP,
        // и штраф за время начисляется на него, а не на шаг физики (T-15).
        requester.TakeActionsBetweenDecisions = false;

        return agent;
    }

    private static void CreateGoal(Transform parent, SoccerArea area, SoccerArea.Side side,
                                   Vector3 localPosition, string tag, string materialName, Color color)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = side == SoccerArea.Side.West ? "GoalWest" : "GoalEast";
        go.tag = tag;
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPosition;
        go.transform.localScale = new Vector3(1f, 2f, 2f * GoalHalfWidth);
        go.GetComponent<Renderer>().sharedMaterial = MakeMaterial(materialName, color);

        var box = go.GetComponent<BoxCollider>();
        box.isTrigger = true;

        var goal = go.AddComponent<SoccerGoal>();
        goal.defendedBy = side;
        goal.area = area;
    }

    private static Transform CreateBox(Transform parent, string name, Vector3 localPosition,
                                       Vector3 scale, string materialName, Color color, string tag)
    {
        var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.name = name;
        box.tag = tag;
        box.transform.SetParent(parent, false);
        box.transform.localPosition = localPosition;
        box.transform.localScale = scale;
        box.GetComponent<Renderer>().sharedMaterial = MakeMaterial(materialName, color);
        return box.transform;
    }

    /// <summary>
    /// Создаёт теги, которых нет в проекте. Теги среды `E08` перечислены
    /// и в <c>ProjectBootstrap</c>, и здесь: сцену можно собрать, не запуская
    /// сначала Tools/RL/Configure Project.
    /// </summary>
    private static void EnsureTags()
    {
        foreach (var tag in new[] { "ball", "playerWest", "playerEast", "goalWest", "goalEast" })
        {
            if (System.Array.IndexOf(UnityEditorInternal.InternalEditorUtility.tags, tag) < 0)
                UnityEditorInternal.InternalEditorUtility.AddTag(tag);
        }
    }

    private static Material MakeMaterial(string name, Color color)
    {
        string path = $"{Root}/Materials/{name}.mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            mat = new Material(shader) { color = color };
            AssetDatabase.CreateAsset(mat, path);
        }
        else
        {
            mat.color = color;
            EditorUtility.SetDirty(mat);
        }
        return mat;
    }

    /// <summary>Рекурсивно создаёт папку ассетов вида "Assets/a/b/c".</summary>
    private static void EnsureFolder(string path)
    {
        var parts = path.Split('/');
        var current = parts[0];
        for (int i = 1; i < parts.Length; i++)
        {
            var next = $"{current}/{parts[i]}";
            if (!AssetDatabase.IsValidFolder(next))
                AssetDatabase.CreateFolder(current, parts[i]);
            current = next;
        }
    }
}
