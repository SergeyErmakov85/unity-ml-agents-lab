# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal RL lab: Unity ML-Agents environments plus a **from-scratch PyTorch training core**.
The point of the project is that training is done by our own code (`python/labrl`), not by
`mlagents-learn` — the stock trainer is kept only as a reference for comparison.

The top-level executable instruction is `CLAUDE_Unity-ml-agents-lab.md` (Russian). It defines
phases, gates, and hard prohibitions; **read it before changing structure**. Work proceeds in
phases with a mandatory stop and user confirmation after each. Current state and open questions
live in `PLAN.md`.

Environment specs ("ТЗ" / TS-NNN) are Russian-language `ENV_SPEC.md` files inside each
environment folder. Follow a spec exactly when one exists. Docs and commit messages are Russian.

## Repository structure

```
docs/          00_AUDIT, 01_STACK, 02_LESSON_MAP, 03_CONVENTIONS, 04_ONNX_CONTRACT,
               05_TENSORBOARD, 06_WORKFLOW, ASSUMPTIONS, adr/, envs/, algos/, templates/
unity/MLAgentsLab/   the single Unity project for all environments
python/        labrl package + tests + .venv (Python 3.10.11)
notebooks/     one training notebook per example
configs/       E##_<Name>__<algo>.yaml (ours) + mlagents/E##_<Name>.yaml (stock trainer)
scripts/       build_env.py, verify_onnx.py, tb.ps1
builds/        headless builds (git-ignored)
results/       runs (git-ignored except README.md)
_archive/      superseded material — nothing is ever deleted outright
```

Unity project layout: `Assets/Shared/` (Scripts/Core, Editor, Materials, Prefabs) and
`Assets/Envs/E##_<Name>/` (Scenes, Scripts, Editor, Prefabs, Materials, Models, ENV_SPEC.md).

Existing environments: `E01_GridWorld` (5×5 grid, tabular Q-learning), `E03_RollerBall`
(roll-to-target). Full map of lessons → examples: `docs/02_LESSON_MAP.md`.

## The one invariant that breaks everything

**Behavior Name in Unity MUST equal the environment id `E##_<Name>`.** It is the only link
between Python and Unity; a mismatch makes Python see no agents, with no useful error.
Enforced in three places: `AgentBase.AssertBehaviorName()` at runtime, `SceneValidator`
before build, `labrl.utils.config` when reading a config.

## Common commands

Everything Python runs from `python/.venv`:

```powershell
$py = ".\python\.venv\Scripts\python.exe"
& $py -m pytest python/tests -q                  # tests
python scripts/build_env.py E03_RollerBall       # headless build (returns Unity's exit code)
python scripts/build_env.py --validate-only      # SceneValidator over all environments
.\scripts\tb.ps1                                 # TensorBoard over results/
```

Scene rebuild (scenes are generated from code, never hand-authored — to change a scene,
change its Setup script and rebuild):

```powershell
& "C:\Program Files\Unity\Hub\Editor\6000.5.8f1\Editor\Unity.exe" -batchmode -quit `
  -projectPath C:\unity-ml-agents-lab\unity\MLAgentsLab `
  -executeMethod GridWorldSetup.BuildScene -logFile build.log
```

Editor entry points: `GridWorldSetup.BuildScene`, `RLEnvironmentSetup.BuildTrainingScene`,
`ProjectBootstrap.ConfigureAndValidate`, `LabRL.EditorTools.BuildScript.BuildEnv`,
`LabRL.EditorTools.SceneValidator.ValidateAllBatch`.

## Architecture

**Unity side.** `Assets/Shared/Scripts/Core`: `AgentBase` (Behavior Name check, episode
metrics), `TrainingAreaBase` (seed/difficulty from `EnvironmentParametersChannel`, one
`System.Random` per arena), `SpawnService`, `MetricsRecorder`. `Assets/Shared/Editor`:
`BuildScript`, `SceneValidator`, `OnnxContractCheck`, `ProjectBootstrap`.

**Python side.** `labrl/envs` (Unity wrappers), `nets` (protocols + MLPs — the user's own
architecture is passed in), `algos` (one file per algorithm, written from scratch, single
`update()` method), `buffers`, `export` (ONNX contract — the critical module), `logging`,
`eval`, `utils`.

**ONNX is the crux.** A plain `torch.onnx.export` produces a graph Unity rejects. The exact
contract (input/output names, constant outputs, opset 9) is in `docs/04_ONNX_CONTRACT.md`,
extracted from package sources. `labrl.export.onnx_verify` blocks acceptance on any mismatch.

## Gotchas

- `com.unity.ml-agents` must be **4.0.3+**: 4.0.0 fails to compile on Unity 6000.5.
- Python must be **3.10.x** — `mlagents-envs` requires `>=3.10.1,<=3.10.12`. The venv is at
  `python/.venv`; `mlagents`/`mlagents-envs` come from git tag `release_23_tag`, not PyPI.
- Never use a Unity/ML-Agents/PyTorch API from memory. Verify against the installed version;
  package sources win over any document (rule 16.4).
- Never delete files — move them to `_archive/<YYYY-MM-DD>/` (rule 16.2).
- Never mark a DoD item done without actually running the check (rule 16.3).
- Never move to the next phase without user confirmation (rule 16.7).
- `.gitattributes` routes binary assets through Git LFS.
- Heuristic control uses the legacy `Input` API; Active Input Handling must stay "Both".
- Scene/prefab/material references rely on `.meta` GUIDs — move assets with their `.meta`.
