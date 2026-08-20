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
python/        labrl package + tests + .venv (Python 3.10.x)
notebooks/     one training notebook per example
configs/       E##_<Name>__<algo>.yaml (ours) + mlagents/E##_<Name>.yaml (stock trainer)
scripts/       build_env.py, train.py, check_inference.py, verify_onnx.py,
               results.py, sync_models.py, tb.ps1
data/demos/    expert demonstrations for E10 (8 KB, committed on purpose)
builds/        headless builds (git-ignored)
results/       runs (git-ignored except README.md)
_archive/      superseded material — nothing is ever deleted outright
```

Unity project layout: `Assets/Shared/` (Scripts/Core, Editor, Materials, Prefabs) and
`Assets/Envs/E##_<Name>/` (Scenes, Scripts, Editor, Prefabs, Materials, Models, ENV_SPEC.md).

Twelve environments, `E00`–`E11`. `E00`–`E07` are **DONE** (trained, ONNX inference verified
in Unity); `E08`–`E11` are **READY_TO_TRAIN** — scene, build, config, notebook and docs are
in place and checked, but no full training run was made. Statuses live in two places that
must agree: `python/labrl/envs/registry.py` and `docs/02_LESSON_MAP.md`.

Full map of lessons → examples: `docs/02_LESSON_MAP.md`. Per-environment cards:
`docs/envs/`. Per-algorithm cards: `docs/algos/`.

## The one invariant that breaks everything

**Behavior Name in Unity MUST equal the environment id `E##_<Name>`.** It is the only link
between Python and Unity; a mismatch makes Python see no agents, with no useful error.
Enforced in three places: `AgentBase.AssertBehaviorName()` at runtime, `SceneValidator`
before build, `labrl.utils.config` when reading a config.

## Common commands

Everything Python runs from `python/.venv`:

```powershell
$py = ".\python\.venv\Scripts\python.exe"
& $py -m pytest python/tests -q                        # tests (301 as of 2026-08-20)
& $py scripts/build_env.py E08_SoccerArena             # headless build (returns Unity's exit code)
& $py scripts/build_env.py --validate-only             # SceneValidator over all environments
& $py scripts/train.py --config configs/<cfg> --quick  # pipeline smoke test, minutes
& $py scripts/train.py --config configs/<cfg> --all-seeds   # full run, 3 seeds
& $py scripts/smoke_all.py                             # smoke every config, one command
& $py scripts/results.py --write docs/RESULTS.md       # regenerate the summary table
.\scripts\tb.ps1                                       # TensorBoard over results/
```

`--quick` cuts the step budget to 1/10 and **does not check the acceptance criterion** —
it verifies the pipeline, not the result. Real failures (exceptions, failed ONNX
verification, missing required tags) still abort the run.

Scene rebuild (scenes are generated from code, never hand-authored — to change a scene,
change its Setup script and rebuild):

```powershell
& "C:\Program Files\Unity\Hub\Editor\6000.5.4f1\Editor\Unity.exe" -batchmode -quit `
  -projectPath C:\unity-ml-agents-lab\unity\MLAgentsLab `
  -executeMethod GridWorldSetup.BuildScene -logFile build.log
```

Editor entry points — one Setup class per environment, all under `Tools/RL/…`:
`GridWorldSetup`, `RLEnvironmentSetup`, `BanditSetup`, `CartPoleSetup`, `BallBalanceSetup`,
`FoodCollectorSetup`, `HunterSetup`, `RacingSetup`, `SoccerSetup`, `MazeSetup`,
`CorridorSetup`, `KeyDoorSetup` — plus `ProjectBootstrap.ConfigureAndValidate`,
`LabRL.EditorTools.BuildScript.BuildEnv`, `LabRL.EditorTools.SceneValidator.ValidateAllBatch`.

## Architecture

**Unity side.** `Assets/Shared/Scripts/Core`: `AgentBase` (Behavior Name check, episode
metrics), `TrainingAreaBase` (seed/difficulty from `EnvironmentParametersChannel`, one
`System.Random` per arena), `SpawnService`, `MetricsRecorder`. `Assets/Shared/Editor`:
`BuildScript`, `SceneValidator`, `OnnxContractCheck`, `ProjectBootstrap`.

**Python side.** `labrl/envs` (Unity wrappers, including `team_env` for two-team scenes
and `demos` for expert data), `nets` (protocols, MLPs, attention, categorical policy,
FCA concept layer — the user's own architecture is passed in), `algos` (one file per
algorithm, written from scratch, single `update()` method), `buffers`, `export` (ONNX
contract — the critical module), `logging`, `eval` (aggregation, success rules, HPO),
`train` (one loop per family: `onpolicy`, `dqn`, `sac`, `selfplay`, `imitation`,
`curriculum`), `utils`.

**Two rules that keep the algorithms honest.** One file per algorithm, and exactly one
`update()` that touches parameters (8.7). When two algorithms share the math — PPO and
`PPODiscrete`, GAIL's policy and `PPODiscrete` — the second **inherits** or **composes**;
it never copies `update()`. Differences that live in the distribution belong to the
network, not the algorithm.

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
- Before setting `MaxStep` and acceptance thresholds, **measure the random-policy
  baseline**. An environment a random policy solves teaches nothing (T-22); one where it
  never sees reward gives a zero gradient (T-23). Both are cheap to detect and expensive
  to miss.
- Anything that transforms the observation must be **inside the ONNX graph** (10.7).
  Violating this leaves ONNX formally valid and silently wrong in Unity. Three places do
  it right: discretisation grid (`E02`), action range mapping (all continuous envs),
  FCA concept layer (`E11`).
- A textbook claim is a hypothesis about your environment, not a fact about it. `E10` was
  built to show BC's distribution shift; measurement showed BC solves it at 100 %. The
  docs were rewritten to match the measurement, not the other way round.
- `.gitattributes` routes binary assets through Git LFS.
- Heuristic control uses the legacy `Input` API; Active Input Handling must stay "Both".
- Scene/prefab/material references rely on `.meta` GUIDs — move assets with their `.meta`.
