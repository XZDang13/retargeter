# retargeter

`retargeter` converts SMPL/SMPL-X human motion into Unitree G1 robot motion through three main interfaces:

- `online`: realtime-style IK retargeting, one frame at a time.
- `refine`: offline training-data preparation, IK retargeting plus Torch refinement.
- `viewer`: replay online or refined outputs with Newton viewers and optional SMPL-X mesh overlay.

Run commands from the repository root with `PYTHONPATH=.`.

## Requirements

Core dependencies:

```bash
python -m pip install numpy scipy pyyaml matplotlib pytest
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
outputs/05_05_refine/retargeted_motion.npz
outputs/05_05_refine/retargeted_meta.yaml
outputs/05_05_refine/retargeted_quality.json
outputs/05_05_refine/human.npz
```

`final_motion.npz` is the refinement training-data product. IK retarget files are kept for audit and comparison. `human.npz` is exported by default for real SMPL/SMPL-X inputs with mesh vertices; use `--no-human-output` to skip it.

Refine fails by default if `RefinementQualityReport.valid` is false. Use `--allow-invalid` to keep invalid outputs for debugging.

`--fps` means input/source FPS override. `--target-fps` resamples SMPL/SMPL-X parameters before FK and controls downstream IK retarget/refinement FPS.

Batch refine runs the same single-clip pipeline for many inputs and writes one standard refine directory per clip. Parallel batch mode is clip-level multiprocessing; each worker creates its own pipeline instance.

```bash
PYTHONPATH=. python -m retargeter.cli.refine \
  --input-dir test_data \
  --input-pattern '*.npz' \
  --recursive \
  --model-type smplx \
  --target-fps 30 \
  --robot g1_29 \
  --refinement-iterations 50 \
  --workers 4 \
  --gpu-ids 0,1 \
  --resume \
  --skip-existing \
  --summary-csv outputs/refine_batch/summary.csv \
  --output outputs/refine_batch
```

Batch inputs can also come from `--inputs` or a newline-separated `--input-list`. Batch outputs are written under `outputs/refine_batch/<input_stem>/` by default; duplicate stems use `__2`, `__3`, and so on. With `--preserve-tree`, files discovered under `--input-dir` keep their relative directory layout under the output root.

The restartable manifest is written incrementally to `outputs/refine_batch/batch_manifest.json`. Use `--dry-run` to write the planned manifest without retargeting, `--summary-csv` for a compact table, and `--fail-fast` to stop after the first blocking failure. By default the CLI continues after item failures and exits nonzero if any item failed or quality-invalid item was not allowed.

### Viewer

Viewer mode replays outputs from online or refine mode. It can take an output directory or a single motion `.npz`.

```bash
PYTHONPATH=. python -m retargeter.cli.viewer \
  --input outputs/05_05_refine \
  --viewer gl \
  --realtime 1
```

Directory input priority is:

```bash
final_motion.npz -> online_motion.npz -> retargeted_motion.npz
```

If the directory contains `human.npz`, viewer uses it automatically as a SMPL-X mesh overlay. You can also pass it explicitly:

```bash
PYTHONPATH=. python -m retargeter.cli.viewer \
  --input outputs/05_05_refine/final_motion.npz \
  --human outputs/05_05_refine/human.npz \
  --viewer gl \
  --human-offset 0,1.25,0 \
  --realtime 1
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
