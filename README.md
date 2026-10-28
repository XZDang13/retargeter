# Retargeter

Human motion (SOMA / SMPL-X) → Unitree G1 retargeting in Swift + Slang. Design: [DESIGN.md](DESIGN.md).

## Build and test

```bash
xmake          # fetches Slang + slang-rhi on first run, builds everything
xmake test     # all test suites (parity, dynamics identities, retargeting, validation)
```

## Generated assets

Some assets are produced by Python tools (tooling only, never on the runtime path). They need a venv
with SOMA-X, usd-core and torch (CPU) under `build/tools/`:

```bash
python3 -m venv build/tools/venv
build/tools/venv/bin/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
build/tools/venv/bin/pip install numpy scipy usd-core==26.3 warp-lang==1.11.1 trimesh rtree
git clone https://github.com/NVlabs/SOMA-X.git build/tools/SOMA-X
build/tools/venv/bin/pip install --no-deps -e build/tools/SOMA-X
```

- Robot meshes for the viewer:

  ```bash
  build/tools/venv/bin/python -I tools/export_robot_meshes.py assets/robots/unitree_g1/g1_29_dof build/viewer/robots/g1_29dof.glb
  build/tools/venv/bin/python -I tools/export_robot_meshes.py assets/robots/unitree_g1/g1_23_dof build/viewer/robots/g1_23dof.glb
  ```

- SOMA body surface (needs the BONES-SEED `soma_shapes/` folder in `assets/body_models/soma/seed_soma_shapes`):

  ```bash
  build/tools/venv/bin/python -I tools/bake_soma_surface.py --soma-x build/tools/SOMA-X/assets \
      --shape assets/body_models/soma/seed_soma_shapes/soma_base_fit_mhr_params.npz \
      --bvh ~/Downloads/soma_uniform/bvh/230322/sweep_floor_side_walk_R_004__A298.bvh \
      --out assets/body_models/soma/surface_lo.npz --fixture build/fixtures/soma_lbs_reference.npz
  ```

BONES-SEED data and anything derived from it (`seed_soma_shapes/`, `surface_*.npz`) is licensed and must
not be committed or redistributed; `.gitignore` excludes it.

## Retarget (realtime solver, batched)

```bash
build/linux/x86_64/release/retarget realtime ~/Downloads/soma_uniform/bvh/230322 --robot g1_29dof --out out --exclude '*_M.bvh'
```

Writes one `.npz` per clip (50 fps by default) plus `out/manifest.jsonl`; rerunning skips finished clips.
`retarget offline …` (same options) runs the whole-trajectory solver instead: stance-foot locking, ground
contact / non-penetration, self-collision and centroidal-dynamics terms, warm-started from the
realtime solution.
Clips are validated (DESIGN §9, rules in `Configs/reject/g1.json`: kinematics, contacts, collisions and
dynamic feasibility — CoM support (unexplained wrench, ZMP), joint velocity and acceleration limits; torques are
recorded, not judged; acceleration limits are derived from the robot model, overridable in `Configs/robots/g1.json`) and written to `out/accepted/` or `out/rejected/`; the npz
also carries the validator's joint torques (`tau`), contact forces (`contact_force`), `dynamics_residual` and
`zmp`. `retarget realtime --help` lists the options.

Validate trajectories from any tool (our `.npz`, NVIDIA soma-retargeter `.csv`) against their source clips:

```bash
build/linux/x86_64/release/retarget validate out --human ~/Downloads/soma_uniform/bvh --report validation.jsonl
```

## Teleop (live streaming)

```bash
build/linux/x86_64/release/retarget teleop --viewer-port 8770      # ZeroMQ server on tcp://127.0.0.1:5560
build/tools/venv/bin/pip install pyzmq                            # client dependency (once)
cd python && ../build/tools/venv/bin/python -m retargeter_client ~/Downloads/soma_uniform/bvh/210531/walk_forward_amateur_001__A001.bvh
```

`--lag 0.04` adds the fixed-lag window: each frame is refined for 40 ms of newer input (stance lock, ground, dynamics,
self-collision, smoothness) before it is sent. The window is solved at 50 Hz (the policy rate; `--lag-rate`) with
5 sweeps (`--lag-sweeps`), answering the frames that came in since together: ~60 ms end-to-end. Its weights are in
the mapping's `fixed_lag` section.
`retarget realtime --lag 0.04 …` evaluates the same on clips.
The client replays BVH clips in real time (one stream per file) and reports latency; open
http://127.0.0.1:8770/ to watch the streams. Use `retargeter_client.TeleopClient` to stream from your own
source (DESIGN §10 for the messages); `retarget teleop --help` lists the server options. End-to-end check:
`build/tools/venv/bin/python python/tests/test_teleop.py`.

## Viewer

```bash
build/linux/x86_64/release/retarget view --port 8080 --motions ~/Downloads/soma_uniform/bvh
```

Open http://127.0.0.1:8080/. Space toggles playback; ←/→ step frames.
