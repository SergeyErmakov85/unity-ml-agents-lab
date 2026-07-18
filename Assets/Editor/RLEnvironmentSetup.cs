using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Собирает готовую сцену для обучения с подкреплением:
/// платформа, агент-шар (RollerAgent + Behavior Parameters + Decision Requester),
/// цель (Target) и ограничивающие стены.
/// Меню: Tools → RL → Build Training Scene
/// </summary>
public static class RLEnvironmentSetup
{
    private const string ScenePath = "Assets/Scenes/RLTrainingScene.unity";

    [MenuItem("Tools/RL/Build Training Scene")]
    public static void BuildTrainingScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---------- Платформа ----------
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.localScale = new Vector3(1f, 1f, 1f); // Plane = 10x10 юнитов
        floor.GetComponent<Renderer>().sharedMaterial = MakeMaterial("FloorMat", new Color(0.35f, 0.35f, 0.4f));

        // ---------- Стены по краям (чтобы шар не укатывался) ----------
        CreateWall("Wall_North", new Vector3(0f, 0.5f, 5.25f), new Vector3(11f, 1f, 0.5f));
        CreateWall("Wall_South", new Vector3(0f, 0.5f, -5.25f), new Vector3(11f, 1f, 0.5f));
        CreateWall("Wall_East",  new Vector3(5.25f, 0.5f, 0f), new Vector3(0.5f, 1f, 11f));
        CreateWall("Wall_West",  new Vector3(-5.25f, 0.5f, 0f), new Vector3(0.5f, 1f, 11f));

        // ---------- Цель (Target) ----------
        var target = GameObject.CreatePrimitive(PrimitiveType.Cube);
        target.name = "Target";
        target.transform.localPosition = new Vector3(3f, 0.5f, 3f);
        target.GetComponent<Renderer>().sharedMaterial = MakeMaterial("TargetMat", new Color(0.2f, 0.8f, 0.3f));

        // ---------- Агент ----------
        var agentGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        agentGo.name = "RollerAgent";
        agentGo.transform.localPosition = new Vector3(0f, 0.5f, 0f);
        agentGo.GetComponent<Renderer>().sharedMaterial = MakeMaterial("AgentMat", new Color(0.2f, 0.5f, 0.9f));

        var rb = agentGo.AddComponent<Rigidbody>();
        rb.mass = 1f;

        // Agent Script
        var agent = agentGo.AddComponent<RollerAgent>();
        agent.target = target.transform;
        agent.MaxStep = 1000;

        // Behavior Parameters: Space Size = 8, Continuous 2 (X, Z)
        var behavior = agentGo.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = "RollerAgent";
        behavior.BrainParameters.VectorObservationSize = 8;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
        behavior.BehaviorType = BehaviorType.Default;

        // Decision Requester: Decision Period = 5
        var requester = agentGo.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = 5;
        requester.TakeActionsBetweenDecisions = true;

        // ---------- Камера ----------
        var cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(0f, 10f, -12f);
            cam.transform.rotation = Quaternion.Euler(40f, 0f, 0f);
        }

        EditorSceneManager.SaveScene(scene, ScenePath);
        Debug.Log($"RL-сцена собрана и сохранена: {ScenePath}");
    }

    private static void CreateWall(string name, Vector3 position, Vector3 scale)
    {
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = name;
        wall.transform.localPosition = position;
        wall.transform.localScale = scale;
        wall.GetComponent<Renderer>().sharedMaterial = MakeMaterial("WallMat", new Color(0.6f, 0.6f, 0.65f));
    }

    private static Material MakeMaterial(string name, Color color)
    {
        string path = $"Assets/Materials/{name}.mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            mat = new Material(shader) { color = color };
            if (!AssetDatabase.IsValidFolder("Assets/Materials"))
                AssetDatabase.CreateFolder("Assets", "Materials");
            AssetDatabase.CreateAsset(mat, path);
        }
        return mat;
    }
}
