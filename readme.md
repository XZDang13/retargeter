# retargeter

`retargeter` converts SMPL/SMPL-X human motion into Unitree G1 robot motion through three main interfaces:

- `online`: realtime-style IK retargeting, one frame at a time.
- `refine`: offline training-data preparation, IK retargeting plus Torch refinement.
- `viewer`: replay online or refined outputs with Newton viewers and optional SMPL-X mesh overlay.

Run commands from the repository root with `PYTHONPATH=.`.

## Requirements

Core dependencies:

```bash
python -m pip install numpy scipy pyyaml matplotlib pytest tqdm
```

Real SMPL/SMPL-X input requires:

```bash
python -m pip install torch smplx
```

Real Newton IK and replay require `newton` and `warp` in the active environment. SMPL-X model files are expected under `assets/body_models/smplx/` unless `--smpl-model-dir` points at another model directory.

## Main Interfaces

### Online Mode

Online mode is the IK-retarget-only path. The Python API is the realtime interface:

```python
from retargeter import OnlineRetargeter

retargeter = OnlineRetargeter(robot="g1_29")
frame_result = retargeter.step(canonical_motion, frame_idx, contact_result=contact)
retargeter.reset()
```

The CLI simulates the online loop over a mock or file input and writes an online artifact layout:

```bash
PYTHONPATH=. python -m retargeter.cli.online \
  --input mock \
  --robot g1_29 \
  --output outputs/online_mock \
  --mock-frames 60
```

Outputs:

```bash
outputs/online_mock/online_motion.npz
outputs/online_mock/online_meta.yaml
outputs/online_mock/online_quality.json
```

Online mode does not run refinement and does not accept refinement options.

### Refine Mode

Refine mode prepares training data. It runs SMPL/SMPL-X preprocessing, Newton IK retargeting, Torch refinement, and refinement quality evaluation.

```bash
PYTHONPATH=. python -m retargeter.cli.refine \
  --input test_data/05_05_stageii.npz \
  --model-type smplx \
  --smpl-model-dir assets/body_models \
  --target-fps 30 \
  --robot g1_29 \
  --refinement-iterations 50 \
  --output outputs/05_05_refine
```

Outputs:

```bash
outputs/05_05_refine/final_motion.npz
outputs/05_05_refine/final_meta.yaml
outputs/05_05_refine/final_quality.json
outputs/05_05_refine/human.npz
```

`final_motion.npz` is the refinement training-data product. IK retarget files are debug output only; pass `--save-retargeted` to write `retargeted_motion.npz`, `retargeted_meta.yaml`, and `retargeted_quality.json`. `human.npz` is exported by default with canonical body motion and contact data; mesh fields are included only when vertices are available. Use `--no-vertices` to omit mesh data and `--no-human-output` to skip `human.npz` entirely.

Refine rejects invalid motions by default instead of exiting with an error. Rejected refined outputs are written under `rejected/`, while top-level `final_motion.npz` is reserved for accepted training data. Use `--allow-invalid` to write invalid final outputs in the normal top-level layout for debugging.

Refinement quality includes a post-refine physical feasibility gate for joint limits, velocity, per-frame robust-percentile acceleration/jerk plus sustained dynamics violations, foot penetration/floating, support availability, and conservative base-of-support diagnostics. Pelvis height is reported but not a default rejection gate. Support availability is on by default, but it rejects only long continuous unsupported intervals, not total unsupported fraction, so jumps and kicks are not rejected solely for contact-light phases. Override thresholds with the `physical_feasibility` section in `--refinement-config`.

`--fps` means input/source FPS override. `--target-fps` resamples SMPL/SMPL-X parameters before FK and controls downstream IK retarget/refinement FPS.

Refine progress uses `tqdm` on stderr. `--progress auto` is the default and shows bars in interactive terminals; use `--progress on` to force progress in logs or `--progress off` for quiet scripting.

Batch refine writes one standard refine directory per clip. Batch execution uses independent worker processes: `--workers N` runs up to `N` clips at the same time, with each worker creating its own pipeline and Newton backend. This is the production batch path for dataset processing; solver-level native microbatching is not used.

```bash
PYTHONPATH=. python -m retargeter.cli.refine \
  --input-dir test_data \
  --recursive \
  --model-type smplx \
  --target-fps 30 \
  --robot g1_29 \
  --refinement-iterations 50 \
  --workers 4 \
  --gpu-ids 0 \
  --resume \
  --skip-existing \
  --summary-csv outputs/refine_batch/summary.csv \
  --output outputs/refine_batch
```

Batch inputs can also come from `--inputs` or a newline-separated `--input-list`. `--input-dir` auto-detects `.npz` files by default and uses a structural motion filter to skip template/calibration files such as CMU `neutral_stagei.npz`: accepted files must look like SMPL/SMPL-X motion sequences with time-series translation and pose arrays. Use `--motion-filter off` to disable this, and use repeatable `--input-pattern` only when you want a different or additional glob. Batch outputs are written under `outputs/refine_batch/<input_stem>/` by default; duplicate stems use `__2`, `__3`, and so on. With `--preserve-tree`, files discovered under `--input-dir` keep their relative directory layout under the output root.

The restartable manifest is written incrementally to `outputs/refine_batch/batch_manifest.json`. Batch mode also writes `outputs/refine_batch/batch_results.csv` with one row per clip and a `decision` column marked `pass` or `reject` for completed items. Use `--dry-run` to write planned `pending` rows without retargeting, `--summary-csv` for an additional compact table, and `--fail-fast` to stop after the first failed item. By default the CLI continues after item failures and exits nonzero only if at least one item failed; quality-invalid items are rejected/recorded without making the batch command fail.

### Viewer

Viewer mode replays outputs from online or refine mode. It can take an output directory or a single motion `.npz`.
Interactive `gl` and `viser` replay at real speed by default; `file` and `usd` export as fast as possible unless `--realtime 1` is passed.

```bash
PYTHONPATH=. python -m retargeter.cli.viewer \
  --input outputs/05_05_refine \
  --viewer gl
```

Directory input priority is:

```bash
final_motion.npz -> online_motion.npz -> retargeted_motion.npz -> rejected/final_motion.npz
```

If the replay directory contains `human.npz`, viewer uses it automatically as a SMPL-X mesh overlay. For rejected refine outputs, `outputs/clip/rejected/final_motion.npz` also autodetects `outputs/clip/human.npz`; if no human file exists, the viewer replays the robot motion only. You can also pass a human file explicitly:

```bash
PYTHONPATH=. python -m retargeter.cli.viewer \
  --input outputs/05_05_refine/final_motion.npz \
  --human outputs/05_05_refine/human.npz \
  --viewer gl \
  --human-offset 0,1.25,0
```

Viewer choices:

- `file`: writes `newton_replay.json`
- `usd`: writes `newton_replay.usd`
- `viser`: starts Newton ViewerViser
- `gl`: starts Newton ViewerGL
- `null`: headless replay smoke test

Viewer only loads and displays motion. It does not run IK, optimize, or modify motion files.

## Output Format

IK retarget-style online outputs contain:

- `fps`
- `robot`
- `joint_names`
- `root_pos_w`: `[T, 3]`
- `root_quat_xyzw`: `[T, 4]`
- `joint_pos`: `[T, D]`
- `joint_vel`: `[T, D]`
- `body_names`
- `body_pos_w`: `[T, B, 3]`
- `body_quat_xyzw`: `[T, B, 4]`
- `success`: `[T]`

Refine `final_motion.npz` contains the same root, joint, and body state arrays plus:

- `root_delta`: `[T, 3]`
- `joint_delta`: `[T, D]`

`human.npz` contains canonical 21-body human motion and, when available, SMPL/SMPL-X mesh fields:

- `vertices_w`: `[T, V, 3]`
- `mesh_faces`: `[F, 3]`
- contact diagnostics for plotting and quality evaluation

Refinement skating uses PHUMA-style soft-contact horizontal foot velocity: each contact point is normalized over positive-contact frames so airborne frames do not dilute the penalty. The default skating weight is tuned for this pipeline's stricter "do not worsen retargeted-motion skating" quality gate and can be overridden with `--refinement-config`.

## Supported Robots

- `unitree_g1_29` / `g1_29`: 29 actuated joints
- `unitree_g1_23` / `g1_23`: 23 actuated joints

Robot specs:

```bash
retargeter/newton/configs/g1_29_robot.yaml
retargeter/newton/configs/g1_23_robot.yaml
```

IK retargeting configs:

```bash
retargeter/newton/configs/g1_29_newton_ik.yaml
retargeter/newton/configs/g1_23_newton_ik.yaml
retargeter/scale/configs/g1_29_scaler.yaml
retargeter/scale/configs/g1_29_ik_targets.yaml
retargeter/scale/configs/g1_23_scaler.yaml
retargeter/scale/configs/g1_23_ik_targets.yaml
```

Newton IK retargeting is single-pass: each frame runs one `full_body_tracking` solve, warm-started
from the previous frame when available. The old coarse-plus-tracking pass schedule is no longer
supported by the solver.

## Debug Preprocessing

The standalone SMPL/SMPL-X preprocessing CLI remains available for debugging:

```bash
PYTHONPATH=. python -m retargeter.cli.preprocess_smpl \
  --input test_data/05_05_stageii.npz \
  --model-type smplx \
  --smpl-model-dir assets/body_models \
  --target-fps 30 \
  --output outputs/05_05_preprocess
```

This is not a main pipeline interface; use `online`, `refine`, or `viewer` for normal workflows.

## Testing

```bash
PYTHONPATH=. pytest -q
```

The tests use mocks where possible. Tests that need real Newton use `pytest.importorskip("newton")`.

## Notes

- Online mode is an IK retarget realtime API; it is not a teleop server or robot upload path.
- Refine mode is offline and intended for training-data preparation.
- Viewer is read-only.
- Quaternion arrays use `xyzw` order.
- The default world frame is `z_up`.
