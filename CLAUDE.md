# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal lab of Unity ML-Agents reinforcement-learning environments. Environments are implemented from detailed Russian-language tech specs ("ТЗ" / TS-NNN) stored as `INSTRUCTIONS.md` inside the environment's folder — the spec's target executor is Claude Code. Follow the spec exactly when one exists (coordinates, reward values, naming, assumptions registry). Commit messages and docs are written in Russian.

## Repository structure

The repo root is a **single Unity project** (Unity 6000.5.4f1, URP, `com.unity.ml-agents` 4.0.3, Input System with Active Input Handling = Both). Each RL environment is a plain asset folder under `Assets/ML-ENVIRONMENTS/<NN-Category>/<EnvironmentName>/` containing its own `Scenes/`, `Scripts/`, `Materials/`, `Editor/` and optionally `config/` (trainer YAML) and `INSTRUCTIONS.md` (spec). Categories run `01-Basics` through `10-Research`. Do NOT create nested Unity projects (own `Packages/`/`ProjectSettings/`) inside `Assets` — everything shares the root project.

Existing environments:
- `Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/` — RollerAgent (roll-to-target), PPO config in `config/RollerAgent.yaml`
- `Assets/ML-ENVIRONMENTS/02-Examples/Greed_world/` — 5×5 GridWorld for tabular Q-learning, spec in `INSTRUCTIONS.md` (TS-001)

`Assets/Editor/ProjectBootstrap.cs` (menu **Tools → RL → Configure Project**, headless method `ProjectBootstrap.ConfigureAndValidate`) idempotently creates/assigns the URP pipeline asset in `Assets/Settings/`, ensures the tags `agent/goal/trap/wall`, sets the build-scene list (auto-discovered from `Assets/ML-ENVIRONMENTS` — do not hardcode scene paths), and opens every environment scene as a smoke test.

`Assets/Editor/MLAgentsTrainingValidator.cs` (menu **Tools → RL → Validate Training Setup**, headless `MLAgentsTrainingValidator.Validate`, also called by `ConfigureAndValidate`) checks every environment is trainable: non-empty Behavior Name, a matching `behaviors:` key in that environment's `config/*.yaml`, project-wide unique behavior names, non-empty action spec, a Decision Requester, and a source of observations. It throws on errors so batch runs exit non-zero.

`Assets/ML-Agents/` holds files vendored from the upstream ml-agents repo (branch `release_23`) with original paths and `.meta` GUIDs preserved, so upstream example environments can be dropped in and resolve their references: `Examples/SharedAssets/` (Scripts, Prefabs, Meshes, Materials) plus two `Examples/WallJump/Materials/*.mat` that the shared prefabs reference. `ModelCarousel.cs` is deliberately excluded (needs `com.unity.recorder`). Upstream materials target the Built-in pipeline and need the Render Pipeline Converter to look right under URP. `Assets/ML-Agents/Timers/` is training output and is git-ignored. Details: `Assets/ML-Agents/README.md`.

## Common commands

PowerShell wrappers in `scripts/` are the normal entry points (they locate `.venv`, Unity, and each environment's config themselves). `scripts/_common.ps1` holds the shared discovery helpers — extend that rather than duplicating path logic.

```powershell
scripts\setup-python.ps1                        # .venv: Python 3.10.12 + mlagents 1.1.0 (via uv)
scripts\train.ps1 -List                         # environments, scenes, configs, behavior names
scripts\train.ps1 <Env> -RunId <id>             # mlagents-learn, then press Play in the editor
scripts\tensorboard.ps1 [-RunId <id>]
scripts\build-scenes.ps1 [<Env>] [-List] [-ValidateOnly]
scripts\new-environment.ps1 -Name <X> -Category <NN-Category>
```

Headless scene rebuild (each environment has an Editor script that programmatically constructs its scene — scenes are generated, not hand-authored; to change a scene, change its setup script and rebuild):

```
"C:\Program Files\Unity\Hub\Editor\6000.5.4f1\Editor\Unity.exe" -batchmode -quit ^
  -projectPath C:/unity-ml-agents-lab -executeMethod GridWorldSetup.BuildScene -logFile build.log
```

- Greed_world: `GridWorldSetup.BuildScene` (menu: Tools/RL/Build GridWorld Scene)
- Hit_the_ball: `RLEnvironmentSetup.BuildTrainingScene` (menu: Tools/RL/Build Training Scene)
- Project config + training-readiness check: `ProjectBootstrap.ConfigureAndValidate`

`scripts/build-scenes.ps1` discovers scene builders by convention: a static editor method whose `[MenuItem]` path starts with `Tools/RL/Build `. Name new builders that way so they are picked up; other `Tools/RL/*` items (themes, screenshots) are skipped in batch runs.

Batch mode requires the Editor to be **closed** — an open Editor holds `Temp/UnityLockfile` and `Unity.exe -batchmode` then exits immediately with code 1 and an empty log. `build-scenes.ps1` detects this and exits 2. With the Editor open, drive it through the `Tools/RL/*` menu items instead (the unity-editor-mcp `menu` tool works for this).

Training without the wrapper (run from repo root, then press Play in the editor):

```
.venv\Scripts\mlagents-learn.exe Assets/ML-ENVIRONMENTS/01-Basics/Hit_the_ball/config/RollerAgent.yaml --run-id=<run-id>
```

Training outputs land in `results/` (git-ignored, as are `Library/`, `Logs/`, `*.onnx.meta`, `.venv/`, `Assets/ML-Agents/Timers/`).

## Python side

`requirements.txt` pins the trainer stack; `requirements.lock.txt` is a full `uv pip freeze` snapshot. Versions that must stay in sync: `com.unity.ml-agents` 4.0.3 and `mlagents` 1.1.0 both speak communication API **1.5.0**. Python must be 3.10.1–3.10.12 (`mlagents` requires it); `.python-version` pins 3.10.12. `setuptools<81` is required — `mlagents/torch_utils` imports `pkg_resources`, removed in 81+.

## Architecture pattern

Every environment follows the same layout:
- `Editor/<Name>Setup.cs` — static editor class with a `[MenuItem("Tools/RL/...")]` method that builds the entire scene from code (materials, prefabs, training area, agent components). All asset paths must stay inside the environment's own folder (use the `Root` const).
- `Scripts/` — the `Agent` subclass (e.g. `GridWorldAgent`, `RollerAgent`) plus separate environment/UI logic (e.g. `GridWorldEnvironment` owns the MDP: cell indexing, rewards, episode termination).
- `config/` — mlagents-learn trainer YAML per behavior name (behavior names must be unique across environments; `MLAgentsTrainingValidator` enforces this).

`tools/env-template/` is the scaffold `scripts/new-environment.ps1` expands (`__ENV__`, `__ENV_LOWER__`, `__BEHAVIOR__`, `__CATEGORY__`, `__ROOT__` placeholders; `.cs.txt`/`.yaml.txt` extensions keep Unity from compiling the template). Keep it in sync when the environment layout changes. `config/ml-agents-reference/` holds upstream's example trainer configs for hyperparameter reference — it is not wired to any environment here.

GridWorld specifics worth knowing: movement is teleport-based (no Rigidbody), rewards/termination are computed from the cell index rather than collisions, and the env supports both external tabular Q-learning (via `CurrentStateIndex`, 0–24) and neural ML-Agents training (one-hot observation). `TrainingArea` is a prefab designed to be instanced K times at 8-unit X offsets for parallel training.

## Gotchas

- `com.unity.ml-agents` must be **4.0.3+**: 4.0.0 fails to compile on Unity 6000.5 (`Match3ActuatorComponent.cs` uses `Object.GetInstanceID()`, which is an obsolete-as-error API there).
- `.gitattributes` routes many binary asset types (textures, models, audio, `.unitypackage`) through Git LFS — ensure `git lfs` is installed before committing such files.
- Heuristic control in agents uses the legacy `Input` API; Active Input Handling must stay "Both" (`activeInputHandler: 2` in ProjectSettings.asset).
- Scene/prefab/material references rely on `.meta` GUIDs — always move assets together with their `.meta` files.
