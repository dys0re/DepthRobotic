# DepthRobotic

Training-free language-guided robotic manipulation with depth-aware perception and closed-loop geometric control.

DepthRobotic combines DepthVLM-4B, GroundingDINO, and EUPE to execute manipulation tasks on LIBERO without training on robot demonstrations. It turns language instructions into ordered skills, grounds objects in camera images, and uses visual feedback and contact measurements to control a Franka Panda arm.

## Capabilities

- **Task planning:** decompose compound instructions into pick-and-place, articulation, pushing, and toggle operations.
- **Object grounding:** resolve spatial relations, refine image regions, read labels, and verify object identity from the wrist camera.
- **Closed-loop grasping:** measure contact height and object width, select grasp strategies, and recover from failed attempts.
- **Placement:** track receiving regions with visual correspondence, verify release, and correct placement errors.
- **Articulation:** locate handles, triangulate across views, operate drawers, and follow door hinge geometry.
- **Evaluation:** enforce action budgets, record trajectories, save incremental results, and compare task outcomes against a fixed reference.

## Architecture

```text
Language instruction
        │
        ▼
    Task planner
        │
        ▼
    Skill dispatch
        │
        ├── Pick and place
        ├── Open and close
        ├── Push
        └── Toggle
        │
        ▼
Visual grounding → Geometric estimation → Contact-aware control
        ▲                                      │
        └──────── Camera and robot feedback ────┘
```

`Agent` coordinates planning and execution. Model adapters implement perception interfaces; `SkillContext` supplies each episode's dependencies and state. The benchmark layer owns initialization, action budgets, success detection, recording, and result persistence.

## Installation

The validated runtime uses Python 3.12, PyTorch 2.10.0 with CUDA, and transformers 5.3.0.

```bash
python -m pip install -e '.[dev]'
```

Install LIBERO and configure its simulation assets separately. Model weights and third-party model source are external dependencies:

| Dependency | Configuration field | Default location |
|---|---|---|
| DepthVLM-4B weights | `model_path` | `weights/DepthVLM-4B` |
| DepthVLM source tree | `source_path` | `vendor/DepthVLM` |
| GroundingDINO weights | `gdino_path` | `weights/grounding-dino-tiny` |
| EUPE source tree | `eupe_repo` | `vendor/EUPE` |
| EUPE-ViT-B checkpoint | `eupe_checkpoint` | `weights/EUPE-ViT-B.pt` |

Set these paths in the YAML configurations before running. Relative paths resolve from the project root. DepthVLM requires its custom Qwen3-VL implementation, including the depth head.

## Running

Run LIBERO-40 with a 2× action budget and live camera feedback:

```bash
python -m depthrobotic \
  --name libero40_2x \
  --cases libero_40 \
  --config configs/exp_budget_relaxed.yaml \
  --baseline configs/exp_live_observation.yaml \
  --change budget_scale
```

Add `--check` to validate configuration and model assets without loading the models. Add `--no-record` to disable video recording.

Supported task selections:

```text
libero_40
libero_90
libero_spatial:0,libero_goal:2
```

The 2× budgets for the four LIBERO-40 suites are 560, 560, 600, and 1040 actions. Initial settling is recorded separately. Each task uses initial state 0 by default, and the grasp-attempt limit scales with the action budget.

Runs write configuration, provenance, per-task events, perception records, and results under `results/<name>/`. Existing run directories are never overwritten. Measured runs require a clean Git working tree; `--allow-dirty` is available for exploratory work.

## Testing

Run tests without model weights or the simulator:

```bash
python -m pytest -q
```

Compare a completed 2× run against the recorded reference:

```bash
python tools/gate.py \
  --run results/libero40_2x/results.json \
  --reference tests/reference/v30_2x_libero40.json \
  --require exact
```

The gate checks population coverage and task outcomes; `exact` also checks action counts. Standard-budget and 2× results are separate measurements. Partial runs do not establish a benchmark score.

## Repository layout

```text
src/depthrobotic/
  agent.py, context.py      Orchestration and dependency interfaces
  config.py, protocol.py   Configuration, populations, and budgets
  planning/                Instruction decomposition
  models/                  Model adapters
  perception/, geometry/   Grounding, correspondence, and geometry
  control/                 Motion and contact primitives
  skills/pick_place/       Approach, grasp, placement, and tool execution
  skills/articulate/       Handle, drawer, and door operations
  skills/simple.py         Pushing and toggle operations
  benchmark/               Environment lifecycle and recording
  results.py, validate.py  Result persistence and integrity checks
  compare.py, diagnose.py  Paired comparisons and failure analysis
tests/reference/           Regression fixtures
tools/                     Evaluation and analysis utilities
```
