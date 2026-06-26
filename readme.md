# retargeter

`retargeter` is an offline Stage 1 motion-retargeting pipeline for converting SMPL/SMPL-X human motion into Unitree G1 joint trajectories with Newton IK.

The current implementation is a practical baseline. It supports preprocessing SMPL motion, scaling human body targets to G1 body targets, solving a two-pass Newton IK problem, clamping joint limits and joint velocities, exporting `.npz` motion files, and replaying the result through Newton viewers.

## Supported Robots

- `unitree_g1_29` / `g1_29`: 29 actuated joints, backed by `assets/robots/unitree_g1/g1_29_dof_rubber_hand/g1_29dof.usda`
- `unitree_g1_23` / `g1_23`: 23 actuated joints, backed by `assets/robots/unitree_g1/g1_23_dof_rubber_hand/g1_23dof.usda`

Robot specs, joint limits, velocity limits, and default poses live in:

```bash
retargeter/newton/configs/g1_29_robot.yaml
retargeter/newton/configs/g1_23_robot.yaml
```

## Repository Layout

```bash
retargeter/
  cli/          # Command-line entry points
  preprocess/   # SMPL/SMPL-X loading, FK, canonicalization, filtering, contacts
  scale/        # Human-to-robot body scaling and IK target construction
  newton/       # Robot specs, Newton backend, Stage 1 solver, export
  visualize/    # Newton replay and diagnostic plots
assets/
  body_models/  # SMPL/SMPL-X model files
  robots/       # Unitree G1 USD assets
tests/          # Unit and smoke tests
test_data/      # Local sample motions
```

## Requirements

Run commands from the repository root with `PYTHONPATH=.`. There is no package metadata file yet, so install the runtime dependencies directly into your environment.

Core dependencies:

```bash
python -m pip install numpy scipy pyyaml matplotlib pytest
```

Real SMPL/SMPL-X input requires:

```bash
python -m pip install torch smplx
```

Real Newton IK and replay require `newton` and `warp` installed in the active environment.

The default SMPL model directory is:

```bash
assets/body_models
```

For SMPL-X, the loader expects model files under either `assets/body_models/smplx/` or a direct `smplx` model directory passed with `--smpl-model-dir`.

## Quick Smoke Test

Use mock human motion to verify the pipeline wiring:

```bash
PYTHONPATH=. python -m retargeter.cli.retarget_stage1 \
  --input mock \
  --robot g1_29 \
  --output outputs/mock_g1_29 \
  --mock-frames 60 \
  --visualize 1 \
  --newton-viewer file
```

Expected outputs:

```bash
outputs/mock_g1_29/motion.npz
outputs/mock_g1_29/meta.yaml
outputs/mock_g1_29/quality.json
outputs/mock_g1_29/newton_replay.json
outputs/mock_g1_29/*.png
```

## Retarget a SMPL-X Motion

For `.npz` motion files:

```bash
PYTHONPATH=. python -m retargeter.cli.retarget_stage1 \
  --input test_data/05_05_stageii.npz \
  --model-type smplx \
  --smpl-model-dir assets/body_models \
  --robot g1_29 \
  --output outputs/05_05_g1_29 \
  --visualize 1 \
  --newton-viewer file
```

For PHUMA-style `.npy` files, pass FPS explicitly:

```bash
PYTHONPATH=. python -m retargeter.cli.retarget_stage1 \
  --input path/to/motion.npy \
  --model-type smplx \
  --fps 30 \
  --smpl-model-dir assets/body_models \
  --robot g1_23 \
  --output outputs/motion_g1_23
```

The `.npz` loader accepts common SMPL/SMPL-X keys:

- Translation: `transl` or `trans`
- Root orientation: `global_orient`, `root_orient`, or `poses[:, 0:3]`
- Body pose: `body_pose`, `pose_body`, or `poses[:, 3:66]`
- FPS: `mocap_frame_rate`, or use `--fps`
- Optional: `betas`, `gender`, hand poses, eye poses, jaw pose, expression

## Pipeline Details

The Stage 1 command performs:

1. Load SMPL/SMPL-X `.npz` or `.npy` motion, or generate `mock` motion.
2. Run SMPL/SMPL-X forward kinematics into a canonical 21-body human representation.
3. Apply low-pass filtering, ground estimation, and foot-contact estimation.
4. Build scaled G1 IK targets from `retargeter/scale/configs/*`.
5. Solve two Newton IK passes, `stage1a` then `stage1b`.
6. Clamp joint limits and frame-to-frame joint velocity.
7. Export motion, metadata, quality, and optional visual diagnostics.

Stage configs:

```bash
retargeter/newton/configs/g1_29_newton_stage1.yaml
retargeter/newton/configs/g1_23_newton_stage1.yaml
```

Scaling and target configs:

```bash
retargeter/scale/configs/g1_29_scaler.yaml
retargeter/scale/configs/g1_29_stage1_targets.yaml
retargeter/scale/configs/g1_23_scaler.yaml
retargeter/scale/configs/g1_23_stage1_targets.yaml
```

## Output Format

`motion.npz` contains:

- `fps`: scalar motion FPS
- `robot`: robot name
- `joint_names`: ordered actuated joint names
- `root_pos_w`: `[T, 3]`
- `root_quat_xyzw`: `[T, 4]`
- `joint_pos`: `[T, D]`
- `joint_vel`: `[T, D]`
- `body_names`: ordered robot body names
- `body_pos_w`: `[T, B, 3]`
- `body_quat_xyzw`: `[T, B, 4]`
- `success`: `[T]` boolean IK success flags

`meta.yaml` records robot, FPS, frame count, joint/body names, config paths, source metadata, and preprocess metadata.

`quality.json` records frame count, success count, success ratio, max absolute joint velocity, and per-frame diagnostics.

## Visualization

Replay an exported Stage 1 motion:

```bash
PYTHONPATH=. python -m retargeter.cli.visualize_stage1 \
  --stage1 outputs/mock_g1_29/motion.npz \
  --robot-spec retargeter/newton/configs/g1_29_robot.yaml \
  --output outputs/mock_g1_29_replay \
  --mode all \
  --viewer file
```

Viewer choices:

- `file`: writes a Newton replay JSON file
- `usd`: writes a USD replay
- `viser`: starts Newton ViewerViser
- `gl`: starts Newton ViewerGL
- `null`: headless replay smoke test

For Viser:

```bash
PYTHONPATH=. python -m retargeter.cli.visualize_stage1 \
  --stage1 outputs/mock_g1_29/motion.npz \
  --robot-spec retargeter/newton/configs/g1_29_robot.yaml \
  --output outputs/viser \
  --mode replay \
  --viewer viser \
  --port 8080 \
  --loop 1 \
  --realtime 1
```

## Testing

Run the test suite from the repository root:

```bash
PYTHONPATH=. pytest -q
```

The tests use mocks where possible. Tests that need the real Newton backend call `pytest.importorskip("newton")`.

## Notes and Limitations

- This project is for offline retargeting. It does not upload motions to a robot or run a live deployment server.
- The current solver is a Stage 1 baseline, not a polished production-quality retargeter.
- Output joint order is defined by the selected robot spec and must be preserved by downstream consumers.
- Quaternion arrays use `xyzw` order.
- The default world frame is `z_up`.
