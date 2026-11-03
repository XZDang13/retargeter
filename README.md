# Retargeter

Human motion (SOMA / SMPL-X) → Unitree G1 retargeting in Swift + Slang. Design: [DESIGN.md](DESIGN.md).

## Setup

Runs on Linux (x86-64 or arm64) with an NVIDIA GPU and CUDA driver, or on macOS with Metal (Apple silicon; Xcode
installed). You need Swift 6 ([swiftly](https://www.swift.org/install/linux/) on Linux, Xcode on macOS) and git.
Everything else is fetched:

```bash
./setup.sh
```

- **Tools.** xmake 3 and [uv](https://docs.astral.sh/uv/) are used from `PATH` when present. Otherwise pinned,
  checksummed builds go to `build/tools/bin`. xmake brings cmake when it is missing, and uv brings Python 3.12 when the
  system has no new enough one. On macOS the Metal Toolchain (an Xcode component) is installed when missing.
- **Build.** The first run downloads Slang and builds slang-rhi with the platform's GPU backend.
- **Venv.** It makes the tooling venv `build/tools/venv` (`tools/requirements.txt`: robot mesh export, the teleop
  client, the Hugging Face CLI) and exports the robot meshes for the viewer.
- **SOMA body surface.** Retargeting, the viewer's body and most tests need `assets/body_models/soma/surface_lo.npz`,
  baked from the BONES-SEED shapes. [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) is gated: accept
  its licence, then log in once with `build/tools/venv/bin/hf auth login` (or set `HF_TOKEN`). `setup.sh` downloads
  `soma_shapes/` (a few MB) into `assets/body_models/soma/seed_soma_shapes` and bakes the surface with numpy, in well
  under a second. SOMA-X's body model (the low-LOD MHR identity model, topology and skeleton transfer, rig) comes as
  fixed arrays in `assets/body_models/soma/soma_x_model.npz` (Apache-2.0, [notice](assets/body_models/soma/SOMA-X_NOTICE.md)),
  so setup needs neither SOMA-X nor torch. Maintainers regenerate those arrays from a pinned SOMA-X with
  `tools/export_soma_x_model.py`, which also checks the numpy bake against SOMA-X itself.
- **Clips.** The retargeting tests and examples read the SOMA uniform BVH clips from `~/Downloads/soma_uniform/bvh`
  (or `SOMA_BVH_ROOT`); `./setup.sh --download-motions [DIR]` fetches the 45 GB archive. Setup itself does not need
  them: without clips the bake checks its alignment on the base rig that comes with the shapes.

BONES-SEED data and anything derived from it (`seed_soma_shapes/`, `surface_*.npz`) is licensed and must not be
committed or redistributed; `.gitignore` excludes it. Re-running skips finished steps; `--help` lists the options.

## Test

```bash
xmake test     # all test suites (parity, dynamics identities, retargeting, validation)
```

The tests read clips from `SOMA_BVH_ROOT` (default `~/Downloads/soma_uniform/bvh`).

Build products land in `build/release/` on every platform. The GPU backend is CUDA on Linux and Metal on macOS
(`--backend cpu` for the CPU one).

## Retarget (realtime solver, batched)

```bash
build/release/retarget realtime ~/Downloads/soma_uniform/bvh/230322 --robot g1_29dof --out out --exclude '*_M.bvh'
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
build/release/retarget validate out --human ~/Downloads/soma_uniform/bvh --report validation.jsonl
```

## Teleop (live streaming)

```bash
build/release/retarget teleop --viewer-port 8770      # ZeroMQ server on tcp://127.0.0.1:5560
build/tools/venv/bin/python -m retargeter_client ~/Downloads/soma_uniform/bvh/210531/walk_forward_amateur_001__A001.bvh
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
build/release/retarget view --port 8080 --motions ~/Downloads/soma_uniform/bvh
```

Open http://127.0.0.1:8080/. Space toggles playback; ←/→ step frames.
