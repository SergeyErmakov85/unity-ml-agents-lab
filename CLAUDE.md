# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal lab of Unity ML-Agents reinforcement-learning environments. Environments are implemented from detailed Russian-language tech specs ("ТЗ" / TS-NNN) stored as `INSTRUCTIONS.md` inside the environment's folder — the spec's target executor is Claude Code. Follow the spec exactly when one exists (coordinates, reward values, naming, assumptions registry). Commit messages and docs are written in Russian.

## Repository structure

The repo root is a **single Unity project** (Unity 6000.5.4f1, URP, `com.unity.ml-agents` 4.0.3, Input System with Active Input Handling = Both). Each RL environment is a plain asset folder under `Assets/ML-ENVIRONMENTS/<NN-Category>/<EnvironmentName>/` containing its own `Scenes/`, `Scripts/`, `Materials/`, `Editor/` and optionally `config/` (trainer YAML) and `INSTRUCTIONS.md` (spec). Categories run `01-Basics` through `10-Research`. Do NOT create nested Unity projects (own `Packages/`/`ProjectSettings/`) inside `Assets` — everything shares the root project.

Existing environments:
- `Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/` — RollerAgent (roll-to-target), PPO config in `config/RollerAgent.yaml`
- `Assets/ML-ENVIRONMENTS/02-Examples/Greed_world/` — 5×5 GridWorld for tabular Q-learning, spec in `INSTRUCTIONS.md` (TS-001)

`Assets/Editor/ProjectBootstrap.cs` (menu **Tools → RL → Configure Project**, headless method `ProjectBootstrap.ConfigureAndValidate`) idempotently creates/assigns the URP pipeline asset in `Assets/Settings/`, ensures the tags `agent/goal/trap/wall`, sets the build-scene list, and opens every environment scene as a smoke test.

## Common commands

Headless scene rebuild (each environment has an Editor script that programmatically constructs its scene — scenes are generated, not hand-authored; to change a scene, change its setup script and rebuild):

```
"C:\Program Files\Unity\Hub\Editor\6000.5.4f1\Editor\Unity.exe" -batchmode -quit ^
  -projectPath C:/unity-ml-agents-lab -executeMethod GridWorldSetup.BuildScene -logFile build.log
```

- Greed_world: `GridWorldSetup.BuildScene` (menu: Tools/RL/Build GridWorld Scene)
- Hit_the_ball: `RLEnvironmentSetup.BuildTrainingScene` (menu: Tools/RL/Build Training Scene)
- Project config check: `ProjectBootstrap.ConfigureAndValidate`

Training (Python `mlagents` package required; run from repo root, then press Play in the editor):

```
mlagents-learn Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/config/RollerAgent.yaml --run-id=<run-id>
```

Training outputs land in `results/` (git-ignored, as are `Library/`, `Logs/`, `*.onnx.meta`, `.venv/`).

## Architecture pattern

Every environment follows the same layout:
- `Editor/<Name>Setup.cs` — static editor class with a `[MenuItem("Tools/RL/...")]` method that builds the entire scene from code (materials, prefabs, training area, agent components). All asset paths must stay inside the environment's own folder (use the `Root` const).
- `Scripts/` — the `Agent` subclass (e.g. `GridWorldAgent`, `RollerAgent`) plus separate environment/UI logic (e.g. `GridWorldEnvironment` owns the MDP: cell indexing, rewards, episode termination).
- `config/` — mlagents-learn trainer YAML per behavior name (behavior names must be unique across environments).

GridWorld specifics worth knowing: movement is teleport-based (no Rigidbody), rewards/termination are computed from the cell index rather than collisions, and the env supports both external tabular Q-learning (via `CurrentStateIndex`, 0–24) and neural ML-Agents training (one-hot observation). `TrainingArea` is a prefab designed to be instanced K times at 8-unit X offsets for parallel training.

## Gotchas

- `com.unity.ml-agents` must be **4.0.3+**: 4.0.0 fails to compile on Unity 6000.5 (`Match3ActuatorComponent.cs` uses `Object.GetInstanceID()`, which is an obsolete-as-error API there).
- `.gitattributes` routes many binary asset types (textures, models, audio, `.unitypackage`) through Git LFS — ensure `git lfs` is installed before committing such files.
- Heuristic control in agents uses the legacy `Input` API; Active Input Handling must stay "Both" (`activeInputHandler: 2` in ProjectSettings.asset).
- Scene/prefab/material references rely on `.meta` GUIDs — always move assets together with their `.meta` files.
