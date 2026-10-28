# Retargeter — Design (v1 draft)

Human motion (SOMA / SMPL-X) → Unitree G1 (23 / 29 DoF) retargeting, written in Swift with
Slang compute kernels. Two solve modes (realtime teleop, offline whole-trajectory), both batched,
both dynamics-aware. Browser viewer for human / robot / overlay replay.

v1 scope: **Linux + CUDA** (plus Slang's CPU target for tests and as a fallback). Metal and other
backends are v2 — the design keeps them possible but does not build them.

---

## 1. Facts about the inputs (verified locally / online)

**SOMA uniform BVH** (`~/Downloads/soma_uniform/bvh/<date>/<name>.bvh`, from the BONES-SEED dataset)
- 142,220 clips, 277 GB; 71,088 are `_M` variants (mirrored twins; same hierarchy).
- One skeleton for every clip ("uniform proportions"): hierarchy is byte-identical across files.
- 78 nodes: `Root` + 77 SOMA joints (body, neck/head/jaw/eyes, full fingers, toes).
- 120 fps (`Frame Time 0.008333`), offsets in **centimetres**, **Y-up**.
- `Root` and `Hips` both carry 6 channels (XYZ position, ZYX Euler); all others 3 rotation channels, ZYX.
- Rest pose is not a T-pose (Kimodo's loader remaps it) — rest-pose alignment is part of mapping config.
- NVIDIA's own `soma-retargeter` (Warp/Newton) targets G1-29 and writes CSV. Useful as a **baseline**
  to compare our metrics against, not a dependency.
- **Body surface**: BVH has no mesh, but **SOMA-X** (NVlabs, Apache-2.0) ships a skinned body for the
  same 78-node skeleton (`SOMA_template_rig.usda`: bind pose, sparse LBS weights, skin LODs of
  18,056 / 4,505 / 612 vertices). BONES-SEED's `soma_shapes/soma_base_fit_mhr_params.npz` gives the
  shape of the uniform skeleton. → We can pose real body vertices from BVH. *Verified in milestone 1:*
  BVH local rotations are SOMA "absolute" rotations (match to 2e-7); the SEED shape fit's bone lengths
  differ from the uniform BVH skeleton by ≤ 5 mm (0.5 cm accumulated at the feet, 1.2 cm at the hands,
  1.5 cm at the head). We skin with BVH FK transforms, so the surface follows the BVH skeleton.

See `docs/retargeting-survey.md` for the survey of existing tools that informed §5a–§9.

**SMPL-X** — model files already in `assets/body_models/smplx/` (neutral/male/female `.npz`).
Input clips are AMASS-style npz (`poses`, `trans`, `betas`, fps).

**Robots** — `assets/robots/unitree_g1/g1_{23,29}_dof/` are USD (converted from MuJoCo MJCF):
- Z-up, metres. `physics.usda` has per-link mass, CoM, diagonal inertia; joint limits (degrees).
- `mujoco.usda` has actuator force ranges (e.g. ±50 N·m) and armature.
- Collision: spheres + cylinders (good for cheap self-collision / ground tests).
- Visual meshes in binary `geometries.usd` (USDC).
- **No joint velocity limits** in the assets → supplied in the robot config from Unitree specs.

## 2. Architecture

```
            ┌───────────── inputs ─────────────┐
 SOMA BVH ──┤ MotionIO: BVH parser, SMPL-X npz  ├─▶ HumanMotion (canonical: metres, Z-up,
 SMPL-X npz ┤ reader, live HumanFrame stream    │    global joint transforms per frame)
 live UDP ──┘                                   │
                                                ▼
                         Mapping (per skeleton × robot, JSON):
                         correspondences, weights, rest offsets, scaling, contact bodies
                                                │ targets per frame
                    ┌───────────────────────────┴──────────────────────┐
                    ▼                                                  ▼
          RealtimeSolver (fixed-lag window)                 OfflineSolver (whole clip)
                    │                                                  │
                    └──────────── shared Slang kernels ────────────────┘
                     FK · costs · autodiff Jacobians · RNEA · contact-wrench QP
                                                │
                                   Validator (metrics, accept/reject)
                                                │
                 ┌──────────────────────────────┼──────────────────────────┐
                 ▼                              ▼                          ▼
           .npz files + manifest         RobotFrame stream           Viewer server ─▶ browser
```

Languages:
- **Swift 6.4** — host code, IO, solvers' outer loops, CLI, viewer server.
- **Slang** — every per-frame / per-batch numeric kernel. `[Differentiable]` cost functions give
  Jacobians via `fwd_diff`, so derivatives are written once.
- **C++ (thin)** — `rtcompute` (`native/`): plain-C API over Slang + slang-rhi, imported by Swift as module
  `CRTCompute` (COM interfaces don't map well to Swift).
- **Build: xmake for everything.** Slang (prebuilt release) and slang-rhi (pinned commit) are xmake packages
  (`xmake/packages.lua`); Swift modules are built whole-module by a custom rule (`xmake/swift.lua`), since
  xmake's built-in Swift support does not emit modules. Tests are executables run by `xmake test`.
- **Python (tooling only)** — one-time USD→glTF mesh export (`usd-core`), and the client package
  that mirrors the stream dataclasses. Not on the runtime path.

## 3. Compute runtime

- slang-rhi device (CUDA in v1; CPU device for parity tests). Swift protocol `ComputeDevice`.
- Kernels compiled at runtime by Slang, cached on disk by (source hash, target, defines).
- Everything is **batch-first**: buffers are laid out `[batch, time, ...]` SoA. Single-stream is batch = 1.
- Realtime path records each solver step into a **CUDA graph** (or a single fused kernel) to keep launch
  overhead out of the latency budget.
- A Swift (double precision) reference of FK, RNEA and costs is the test oracle for the Slang kernels.
- *As built:* the device runs the solvers only. Skeleton FK, skinning and contact analysis run on the CPU
  (parallel over clips and frames), and the batch command overlaps them across batches: while the device solves
  batch k, batch k + 1 is loaded and prepared and batch k - 1 validated and exported (§14, throughput).

## 4. Models

**Robot model** (`RobotModel`, built from USD + `robot.json`):
- Kinematic tree in topological order: parent, joint axis, fixed transform, limits (pos/vel/effort),
  armature, mass/CoM/inertia, collision primitives, foot contact geometry (sole rectangle corners).
- Floating base = pelvis; configuration q = (p ∈ R³, quat, joints). Updates in tangent space.
- Canonical joint order = Unitree SDK motor order (documented per robot; stored in every output).
- We parse only what we need from `.usda` text (prims, xforms, physics attrs). Meshes come from
  the one-time glTF export.

**Human model** (`HumanSkeleton` + `HumanMotion`):
- Skeleton: joint names, parents, rest offsets. Motion: root translation + local rotations per frame,
  plus cached global transforms.
- Loaders normalise to metres, Z-up, fixed forward axis. BVH parse is a hot path for 142k clips →
  fast hand-written parser + optional binary cache.
- SMPL-X: joints from `J_regressor`·shape for given betas; body joints only (no hands in v1).
- The solvers never see a specific skeleton — only "target bodies" produced by the mapping.

**Body surface proxy** (`BodySurface`, used for ground and contact detection, §5a):
- A low-LOD skinned vertex set + LBS weights + per-vertex **contact group** label, posed per frame by an
  LBS kernel from the global joint transforms.
- SOMA: 4,505-vertex LOD baked once from SOMA-X + BONES-SEED shape (`tools/bake_soma_surface.py` →
  `assets/body_models/soma/surface.npz`). SMPL-X: native LBS from the local model npz (10,475 vertices,
  or a decimated subset); groups from the Meshcapade SMPL-X part segmentation.
- Fallback when no mesh is available: capsules/spheres per bone (sole box, heel, toe, knee, hand,
  pelvis, back), radii from anthropometrics.
- Contact groups: L/R heel, L/R toe-ball, L/R knee, L/R hand, L/R elbow, pelvis/buttocks, back, head.

## 5. Mapping config (JSON, per skeleton × robot)

```jsonc
{
  "skeleton": "soma77", "robot": "g1_29dof",
  "scale": { "mode": "per_limb" },  // per_limb (default: per-segment factors, root scaled uniformly) | uniform_height | none
  "rest_alignment": { "LeftArm": [qw,qx,qy,qz], ... },
  "targets": [
    { "human": "Hips",      "robot": "pelvis",            "w_pos": 1.0, "w_rot": 0.5 },
    { "human": "LeftFoot",  "robot": "left_ankle_roll_link", "w_pos": 5.0, "w_rot": 1.0, "contact": "foot_left" },
    { "human": "LeftHand",  "robot": "left_wrist_yaw_link", "w_pos": 2.0, "w_rot": 0.5 }
    // ...
  ]
}
```
Shipped: `soma77 × g1_23`, `soma77 × g1_29`, `smplx × g1_23`, `smplx × g1_29`.

Also in the mapping: `contacts` — which human contact groups map to which robot contact patches
(e.g. heel/toe → sole corners, knee → knee sphere, hand → wrist/hand sphere, pelvis → pelvis box).

Lessons from existing tools (survey §2): scale per segment but keep the **root uniformly scaled**
(otherwise foot sliding); keep hand-authored rest-pose offsets per link; stiffer priors on yaw joints.

As built (2026-10-26): `scale.mode = per_chain` (each target scaled from the root by its chain-length ratio, the
root uniformly by pelvis height) plus **limb anchors** (`scale.anchors`). The whole-body audit
(`SCALE_DIAG` in Tests/RetargetTests) showed chain ratios from the root misplace the G1's limbs in the reference
pose: shoulders 4 cm too narrow and 3 cm too low (spine-dominated ratio 0.59, hands into the G1's wide hip
housings), hip goals 2 cm too wide (ratio 1.36), feet 3 cm too narrow and 1.5–3 cm out of reach. An anchor
(LeftArm/RightArm, LeftLeg/RightLeg) is placed at the robot's own link position in the reference pose, carried by
its parent joint's rotation (chest, pelvis); the limb hangs from it scaled by the straight-line anchor-to-target
distance ratio of the two reference poses (summed link lengths overstate the G1's zig-zag hip chain). For the legs
only the height hangs from the hips; their horizontal placement stays uniformly scaled from the pelvis
(`"horizontal": "uniform"`), because uniform scaling keeps a stance foot under the body as the person's was
(anchoring it at the G1's wider hips moved stance feet ~3 cm out and failed the ZMP rule on walking). Result in the
reference pose: anchors exact, knees/ankles 2 mm, toes 7 mm, hands 2.6 cm (the G1's forearm offset); on the 10
comparison clips offline tracking 5.1 → 3.7 cm and 7/10 accepted.
Weights are tuned by `retarget tune` on a fixed clip corpus with p95 / worst-clip guards, not by eye.

## 5a. Ground estimation and body-contact detection

Runs on the **human** side, from posed body-surface vertices (§4), before any solve. Per frame and per
contact group: `h` = height of the group's lowest-k vertices (k ≈ 8), `v` = their horizontal speed
(zero-phase central difference offline; backward difference + one-pole ~10 Hz filter in realtime).

**Ground (offline, whole clip):**
1. Collect `h` of foot groups on quasi-static frames (`v` < 0.2 m/s); majority / mode vote with ±2 cm
   band → ground height `g` (lowest dense mode). Shift the clip so `g` = 0 — one shift per clip, never
   per frame (per-frame shifts cause jitter).
2. Robust plane fit (RANSAC, 1.5 cm inliers) over static contact points → report tilt; > 2° is flagged.
3. Other height modes > 4 cm above `g` holding static foot contact > 0.25 s → `env_support`
   (stairs, platform, box).
4. Fallback when no static frames exist: 1st percentile of all-vertex heights.

**Ground (realtime):** initialise from the median static foot height over a ~1 s calibration window,
then track with a slow EMA (τ ≈ 5 s) only on frames with both feet static within ±2 cm; upward drift
capped at 1 cm/s so stepping onto a box does not drag the ground.

**Contacts (all groups, hysteresis):**

| group           | enter: h − g / speed   | exit: h − g / speed (2 frames) |
|-----------------|------------------------|--------------------------------|
| heel, toe       | < 3 cm and < 0.15 m/s  | > 5 cm or > 0.3 m/s            |
| knee, hand, elbow | < 4 cm and < 0.25 m/s | > 8 cm or > 0.4 m/s           |
| pelvis, back, head | < 5 cm and < 0.25 m/s | > 8 cm or > 0.4 m/s          |

Gaps ≤ 3 frames are bridged, runs < 50 ms dropped. Offline uses look-ahead (lift-off rollback);
realtime commits contact changes 2–3 frames late for debouncing (≈ 20 ms at 120 Hz, inside budget).

**Support and environment** (as implemented in milestone 1b, `Sources/Contact/`):
- CoM = body-part group centroids weighted by de Leva mass fractions (the plain vertex centroid is
  biased: the head + hair is 25 % of the vertices). ZMP of those parts as point masses,
  p = Σ m_i [(g + z̈_i) r_i − z_i r̈_i] / Σ m_i (g + z̈_i), on smoothed centroids (2026-10-28; the CoM-only
  cart-table model ignored swinging limbs and a twisting torso and put the ZMP 20-40 cm outside the feet in
  turning jogs). Ballistic frames (g + a_z < 0.25 g) are `flight`.
- Support hull = contact patches (every vertex of a contacting group within 3 cm of its lowest point),
  not just the lowest-k points — single-support walking needs the whole sole.
- ZMP > 6 cm outside the hull for ≥ 0.3 s → `unsupported`. The hull takes the contacts and any part within its
  enter height of the ground however fast (a sliding foot carries weight). Each unsupported frame is explained
  greedily: add the elevated (feet > 4 cm, others > 6 cm), static (< 0.1 m/s horizontally and vertically) part
  whose patch brings the ZMP closest to the hull, until it is inside. Only parts free below count: a part whose
  contact patch mostly has another body part within 4 cm horizontally and > 5 cm lower (a torso above the
  legs, a hanging arm, the head) cannot rest on anything. Then every static elevated part whose patch comes within
  15 cm of a chosen one, at most 2 cm above and 8 cm below it, rests on the same surface (pelvis, thighs and knees
  on a seat; the greedy alone picked the pelvis only). Offline, a labelled part keeps its label for its whole rest
  (the run of elevated static frames), also where the feet alone would carry the body. Feet → `env_support` (stairs, platform), others → `env_contact`
  (chair → pelvis). Unexplained runs stay `unsupported`.
- Penetration (source-data flag): feet > 3 cm, limbs/trunk > 8 cm below the local floor (median static foot
  height within 0.5 m) for ≥ 25 ms.
  **Hands are not checked**: SEED finger capture bends through the floor under load (push-up: wrist at
  floor level, fingertips 8.5 cm below; lying: up to 19 cm). Hand *contacts* are still detected; the wrist
  end (distal 15 %) of the forearm belongs to the hand group.
- Causal mode additionally flags `env_support` when static feet hold > 4 cm off the tracked ground for
  ≥ 0.25 s (a realtime tracker cannot see a lower floor that comes later, e.g. walking down stairs).

Groups (21): L/R heel, toe, knee (proximal 35 % of the shin), shin, thigh, hand, elbow (proximal 30 % of
the forearm), forearm, upper arm; pelvis, torso, head — from each vertex's dominant skinning joint.

**How contacts are used:**
- Ground contacts of any group that the mapping maps to a robot patch become **robot contacts** in the
  solvers (stance lock, non-penetration, contact forces in the dynamics). So kneeling, crawling and
  sitting on the floor are representable, not auto-rejected.
- `env_contact` / `env_support` → offline auto-reject in v1 (no scene model). Realtime only flags them.
- Contact masks and ground height are stored in the output and drawn in the viewer.

## 6. Shared cost / constraint library (Slang)

Kinematic:
- target position / orientation tracking (robust loss, e.g. Huber)
- joint position limits (barrier), velocity / acceleration limits
- posture regulariser toward a nominal pose
- joint limits at 98 % of range during the solve with a smooth inner-limit penalty; hard clamp on output
- contact (any mapped patch): no-slip while in contact, flat sole for feet, ground non-penetration (z ≥ 0)
- self-collision (sphere/cylinder pairs, pre-filtered pair list)
- offline only: hard constraints (non-penetration, stance lock, self-collision, limits) via augmented
  Lagrangian on top of the LM solve — soft penalties alone leave ~1 cm residuals (survey §2)

Dynamics:
- **Centroidal consistency**: m·c̈ = Σfᵢ + m·g and L̇ = Σ(pᵢ − c)×fᵢ with contact forces fᵢ at the
  corners of each active contact patch, fᵢ inside linearised friction cones, fᵢ = 0 when not in contact.
- **Joint torques** from floating-base RNEA (as built: recorded and exported, not judged, not in the solver).
- ZMP inside support polygon with margin (falls out of the above; also reported as a metric).

As implemented (milestone 5): the validator runs the full model — double-precision floating-base RNEA
(`Sources/Dynamics`) and per-frame contact forces by non-negative least squares on friction-pyramid edges
(μ = 0.7) at the robot's own ground-touching surface points, torque rows only breaking ties between force
distributions. The offline solver uses point-mass links: centroidal wrench rows (contact forces re-solved in
the kernel every step, warm-started from the previous sweep). The solver's torque rows were removed on
2026-10-26: torques are not judged and the term changed no judged metric (see §13, milestone 5).

Contacts in v1: **any ground contact detected in §5a** that the mapping maps to a robot patch (feet,
knees, hands, pelvis …). Support from the environment (chairs, tables, walls, stairs) is not modelled
in v1 and leads to rejection.

## 7. Realtime mode (teleop)

- Budget: **≤ 100 ms end-to-end** (input received → RobotFrame emitted). This is generous: the solve
  itself targets < 5 ms per batch on GPU, so we spend the slack on quality.
- **Fixed-lag smoother**: keep a sliding window of the last W frames (default ~0.3 s), re-optimise the
  window each tick with warm start, and emit the frame `lag` behind the newest input
  (default lag 40 ms, configurable 0–80 ms). lag = 0 gives a purely causal solver.
- Window costs = full kinematic set + centroidal dynamics (contact forces as window variables).
  *Measured in milestone 5:* a causal (lag = 0) centroidal penalty on backward-difference accelerations makes
  the realtime results worse at every weight tried (correcting frame t moves the error to t + K) and diverges
  at weight 1, so it ships disabled (`solver.dynamics = 0`); it needs the fixed-lag window.
- Ground + contacts: causal versions of §5a.
- Targets are **pelvis-relative**; option to track the upper body by rotation only and the lower body by
  position + rotation (TWIST2 style) — robust to global drift in VR / mocap streams.
- No heavy output filtering (a 14-tap moving average costs ~0.25 s of lag); smoothness comes from the window.
- Safety layer on output: clamp to pos/vel limits relative to the last emitted frame; if a step is
  infeasible, flag the frame (`flags` field) and hold/blend instead of jumping. Realtime never rejects
  — it degrades and reports.
- **Batch**: B independent streams solved together (multiple operators / robots / sim envs).
  Same kernels, batch dimension = stream.
- *As built (M6 phase A, lag 0):* `LiveRetargeter` (`Sources/Retarget/RealtimeStreams.swift`) steps the
  realtime kernel one frame per stream per tick, the solver state of every stream (root, joints, CoM history)
  kept on the device between ticks; the CPU stage per frame is the batch path's (skeleton FK, body skinning,
  `CausalContactTracker`, goals; 0.3 ms). A streamed clip reproduces the batch realtime solve bit for bit
  (test). Safety layer: each joint's step is limited to its velocity limit (`Configs/robots/g1.json`) relative
  to the last emitted pose, `dof_vel` is the backward difference of the emitted poses, and `flags` report
  clamped / cold start / unsupported / penetration / environment support / flight / frames dropped. Server
  (`retarget teleop`): frames of all waiting streams are solved together in arrival order; a stream more than
  `--max-backlog` (0.25 s) behind drops its oldest frames. Measured (RTX 3090 shared with another GPU job,
  120 Hz input, localhost): end-to-end (client send → robot frame back) 1.9 ms median / 3.1 ms p95 for one
  stream, 2.2–2.5 / ≤ 4.6 ms for four; server solve 1.7 ms per tick. Two fixes made per-frame dispatches cheap:
  the kernel's sparsity structure (column masks, pair list, minimum-degree order, factor lists) is computed on
  the host once per weight set (`RetargetSolver.factorStructure`, same algorithm; it took ~3 ms per dispatch on
  one GPU lane), and buffer uploads on CUDA are one `cuMemcpyHtoDAsync` on the queue's stream instead of
  slang-rhi's encoded upload (~1 ms each, 12 per dispatch). Both leave every output bit-identical.
- *As built (M6 phase B, fixed lag):* `FixedLagSolver` (`Sources/Retarget/FixedLagSolver.swift`) on the offline
  kernel, not a separate window kernel. Each tick appends the newest frame of every stream and runs `--lag-sweeps`
  symmetric Gauss-Seidel sweeps of the offline costs over the frames not yet emitted (the last `lag` s), then emits
  the frame leaving the window; emitted frames are never revised and serve as fixed context (4 frames) for the
  smoothness stencils. One dispatch per tick (`sweepWindow`: one group per stream loops over its sweeps and frames);
  a dispatch per frame solve cost ~1.6 ms each (106 ms per tick) against ~0.3 ms of work. Differences from offline,
  each measured on the comparison clips:
  - A new frame starts from the previous frame's solution moved by the realtime solve's change. Started at the
    realtime pose (which locks no feet and drifts from the window's), jerk on walk / sweep ended above lag 0's.
  - Stance anchors: mean sole position *of the realtime solve* over the run so far, frozen when the run's first
    frame is emitted. Anchors from the window's own poses held a lagging robot where it was (tracking 0.3-0.8 m).
  - The dynamics term at a third of the offline weight, held by a reference prior, on lagged contacts (2026-10-27,
    second round; below). Without the prior, with 40-200 ms of lookahead, it traded a little tracking now for less
    unexplained wrench, frame after frame, and the emitted frames kept it: the robot sank 0.4 m within a second on a
    sitting clip and drifted 0.8 m on a walk. (The whole-trajectory solve does not: falling costs tracking over the
    rest of the clip.) The kernel skips the CoM kinematics and the contact-force solve at weight 0.
  - Contact and lock ramps only ramp out at run ends already seen.
  `retarget realtime --lag S [--lag-sweeps N]` runs clips through it as live streams (`LaggedRetargeter`);
  `LiveRetargeter` / `retarget teleop --lag S` use the same solver, and a test checks that the two agree bit for
  bit. Comparison clips without the two human-stage rejects (sit on chair, stairs), 2026-10-27:

  | 8 clips                   | lag 0  | 40 ms, 3 sweeps | 40 ms, 10 | 100 ms, 3 | offline |
  |---------------------------|--------|-----------------|-----------|-----------|---------|
  | accepted                  | 0      | 4               | 4         | 4         | 7       |
  | tracking pos mean (m)     | 0.023  | 0.052           | 0.052     | 0.051     | 0.039   |
  | tracking rot p95 (deg)    | 16.4   | 26.8            | 31.0      | 26.7      | 19.4    |
  | foot skate excess (m/s)   | 0.016  | 0.030           | 0.009     | 0.020     | 0.006   |
  | ground penetration (m)    | 0.066  | 0.0024          | 0.0015    | 0.0020    | 0.0028  |
  | self-collision (m)        | 0.026  | 0.0004          | 0.0005    | 0.0002    | 0.0014  |
  | joint jerk p99 (rad/s³)   | 2398   | 1361            | 995       | 1170      | 1235    |
  | dyn residual p95 (m·g)    | 0.159  | 0.182           | 0.182     | 0.170     | 0.043   |
  | ZMP outside (m)           | 0.047  | 0.008           | 0.010     | 0.009     | 0.010   |

  Contacts, collisions and smoothness reach offline level; tracking (esp. rotation) and the dynamics residual do
  not. Rejects at 40 ms / 3 sweeps: jump and stand-up (dynamics residual, foot skate), walk (foot skate), kneeling
  (tracking, as offline). The stairs clip diverges (tracking 0.39 m) where lag 0 does not.
  Speed (RTX 3090 shared with another process at 100 % GPU load): a window solve every frame (40 ms / 3 sweeps,
  ~7 ms + the realtime step) did not keep up with 120 Hz input. Not helped: 256 threads per group, a single-lane
  backward substitution (both tried, same output, no faster).
  *Window at the policy rate (2026-10-27):* the robot policy runs at 50 Hz, so the window is solved at 50 Hz of
  stream time (`--lag-rate`, default 50; a tick wherever the stream's frame time crosses a 20 ms boundary), and each
  solve emits every frame that is `lag` old by then; the solve itself stays at the source rate (the offline weights
  are per source frame). In lag mode the server hands all waiting frames of a stream to the engine at once, so their
  realtime solves share one dispatch; the window still takes them one at a time, so it solves at the same frames
  however they were batched (live = batch simulation, bit for bit, tested with batches of 1-4 frames).
  Within the 20 ms tick, one stream affords 6 sweeps at lag 40 ms (~13 ms per window solve; 10 sweeps took ~20 ms).
  Default `--lag 0.04 --lag-sweeps 6 --lag-rate 50`: 3/8 comparison clips accepted (crawl, jump, stand-up, walk and
  kneeling rejected; tracking 0.052 m / 24.9 deg, foot skate 0.040, penetration 0.0025 m, self-collision 0.0009 m,
  jerk p99 1576, ZMP outside 0.006 m). Live, 120 Hz input, one or two streams: end-to-end 65 ms median, 76 ms p95,
  77 ms max (lag 40 + waiting for the next tick 0-20 + ~10 compute), no dropped frames. Also measured at 50 Hz:
  60 ms / 4 sweeps 2/8, 100 ms / 3 sweeps 3/8 (no better than 40 ms / 6). Starting new frames part of the way
  towards the realtime pose (10 %, 30 %) improved tracking to 0.042 m / 20 deg but brought back penetration,
  collisions and jerk, and accepted 2/8: not used.
  *Second round (2026-10-27): drift, contacts, dynamics.* Diagnosis on the 50 Hz window (lag 40 ms):
  - Not a time lag (best alignment with offline within one 20 ms frame) but accumulated offsets. Arms: the
    self-collision push ratcheted the arms along their null space (left shoulder yaw 0.08 -> 0.45 rad from offline
    over body check; with collisions off it stayed within 0.02 rad of the realtime solve), carried on by each new
    frame's starting point. Root: it trailed the realtime solve by 65-155 ms; the stance lock held it (lock off: 0.3
    ms on walk) as the anchors average where the realtime foot was early in the stance. Neither a lower root
    acceleration weight nor blending only the root position towards the realtime pose helped.
  - Reference prior (`fixed_lag.reference` 1 for joints and root rotation, `reference_root` 3 for the root
    position): each frame pulled to its realtime solve, an absolute anchor, so nothing integrates from frame to
    frame. On two 3-minute dance clips the window stays within 0.07-0.09 m of the realtime root.
  - Contacts decide foot skate: the window locked feet on the causal tracker's flags, which change ~45 ms late, while
    the validator judges skate where the offline contacts say the foot is planted. With the offline contacts as an
    oracle, skate excess 0.039 -> 0.014 (comparison clips) and 0.30 -> 0 (dance clip). `LaggedContactEstimator`
    (`Sources/Contact`) gives the window flags with the offline test, decided 2 frames late (centred speed over
    ±20 ms, the same hysteresis, the causal ground, no run cleaning): 0.0185 and 0.009. A new frame starts with the
    last decided flags.
  - Dynamics term: weight 3 (offline) diverged on long clips (walk fell after ~17 s) even with the prior; dropping
    the vertical force row did not help; anchors at the robot's own touchdown positions drifted 0.64 m on a dance
    clip. With lagged contacts, weight 1 is the best (0.5: 4/8, 1: 6/8, 2: tracking 0.067, 3: diverges; this
    bracket measured with the vertical row off) and stays bounded on the 3-minute clips. Its contact points are the
    current solution's locked sole corners and patches, as offline.

  | 8 clips, lag 40 ms, 50 Hz  | lag 0  | first version (6 sweeps) | final (5 sweeps) | offline |
  |----------------------------|--------|--------------------------|------------------|---------|
  | accepted                   | 0      | 3                        | 6                | 7       |
  | tracking pos mean (m)      | 0.023  | 0.052                    | 0.051            | 0.039   |
  | tracking rot p95 (deg)     | 16.4   | 24.9                     | 25.0             | 19.4    |
  | foot skate excess (m/s)    | 0.016  | 0.040                    | 0.014            | 0.006   |
  | ground penetration (m)     | 0.066  | 0.0025                   | 0.0028           | 0.0028  |
  | self-collision (m)         | 0.026  | 0.0009                   | 0.0009           | 0.0014  |
  | joint jerk p99 (rad/s³)    | 2398   | 1576                     | 1540             | 1235    |
  | dyn residual p95 (m·g)     | 0.159  | 0.153                    | 0.083            | 0.043   |
  | ZMP outside (m)            | 0.047  | 0.006                    | 0.006            | 0.010   |

  Final: lagged contacts, prior, dynamics 1, 5 sweeps (`--lag-sweeps`, default 5). Rejected: jump (dynamics
  residual) and kneeling (tracking, as offline). Tracking stays ~1.2 cm / 5.6 deg behind offline. Live on the
  free GPU (the earlier timings had another process at 100 % GPU load, about 2x slower): one stream 58 ms median,
  67 ms p95, 68 ms max end-to-end; four streams ~60 / 70 ms; ~5 ms of server work per step.

## 8. Offline mode (quality)

Unknowns per clip: qₜ and contact forces fₜ for all frames. Pipeline:
1. **Preprocess**: load, normalise, keep source rate for the solve (120 Hz for SOMA), export at
   `--fps` (**default 50**); zero-phase Butterworth pre-filter (3 Hz root, 6 Hz pose);
   ground + contacts per §5a (non-causal, so much cleaner than realtime); human-side pre-filter (§9).
2. **Init**: run the realtime solver over the clip (lag = window) as the initial guess, with a
   weight schedule (orientations + end-effectors first, then all positions). *As built:* lag 0, with long
   clips cut into 4 s segments (each started 1 s early from a cold start, those frames dropped;
   `warm_segment`, `warm_overlap`) solved side by side, since the causal solve is sequential per stream.
3. **Kinematic stage**: whole-trajectory Gauss-Newton / LM. Finite-difference smoothness terms couple
   neighbouring frames → normal equations are **block-banded**. GPU evaluates residuals + Jacobians for
   all frames in parallel; banded system solved with block-banded Cholesky (CPU first; GPU cyclic
   reduction later if it is the bottleneck).
4. **Dynamics stage**: add centroidal terms and contact-force variables; continue LM from stage 3.
   Same banded structure (forces are per-frame blocks). *As built:* not a separate stage — the dynamics rows
   are part of every Gauss-Seidel sweep (with the same continuation ramp), and the contact forces are
   eliminated per frame by a non-negative least-squares solve inside the kernel instead of being LM variables.
5. **Validate** (§9) → accept / reject.
- **Batch**: clips bucketed by length, padded and masked, solved as one batched problem per bucket.
  Job runner over a directory or file list; resumable (skip clips whose output + manifest row exist);
  multi-GPU later.

## 9. Validation and auto-reject

One `Validator` computes metrics for any robot trajectory (ours, NVIDIA's baseline CSVs, imports).
Thresholds live in `reject.json` (per robot); every metric is stored whether or not it fails.
Rejection runs in two stages: a cheap **human-side pre-filter** (rejects before spending solve time) and a
**robot-side post-filter**. Defaults below are published values where they exist (survey §5) and must be
calibrated on SOMA.

Human-side pre-filter:

| metric                                                    | default threshold        | source        |
|-----------------------------------------------------------|--------------------------|---------------|
| `env_contact` / `env_support` (§5a)                       | any                      | ours          |
| unexplained unsupported time (§5a)                        | > 0.5 s (warn > 0)       | ours          |
| human penetration below ground                            | > 2 cm for > 3 frames    | ours          |
| root jerk                                                 | > 50 m/s³                | PHUMA         |
| CoM–CoP / support distance                                | > 0.1 m for > 100 frames, or at first/last frame | PBHC |
| clip length                                               | < 10 frames              | UniTracker    |
| input sanity: NaNs, lost tracking, root teleports         | any                      | —             |

Robot-side post-filter:

| group     | metric                                                      | default threshold   |
|-----------|-------------------------------------------------------------|---------------------|
| tracking  | mean / p95 target position error (after scaling)            | 5 cm / 10 cm        |
| tracking  | end-effector max error; link orientation error              | 15 cm; 10°          |
| limits    | joint position / velocity beyond limits (frames)            | 0                   |
| smoothness| p99 joint acceleration / jerk; joint-jump detector          | per joint           |
| contact   | stance slide speed; floating / penetration in contact       | 10 cm/s; 1 cm       |
| collision | self-collision penetration                                  | 0.5 cm              |
| dynamics  | contact-wrench residual (unexplained base wrench), N / N·m  | relative to m·g     |
| limits    | joint acceleration / effort-limited acceleration, held 50 ms | ≤ 1.0×             |
| dynamics  | torque over effort limit, held 50 ms                        | recorded, not judged |
| dynamics  | CoM / ZMP stability margin below 0, longest run             | 50 ms               |
| boundary  | first frame reachable from default pose; last frame statically stable | required  |

- **Auto-reject is ON by default** (`--auto-reject` / `--no-auto-reject`).
- v1 rejects **whole clips** only — no trimming or splitting of bad segments.
- Rejected clips are **still written** (status = `rejected`, reasons + offending frame ranges), into
  `rejected/` beside `accepted/`, so they can be inspected in the viewer. `--drop-rejected` skips writing.
- `manifest.jsonl`: one row per clip — source, robot, status, reasons, metrics, config hash, timings.
- A clip that crashes or fails to load is recorded as `error` in the manifest; it never aborts the batch.

Out of scope for v1: **simulation tracking check** (replay through a PD-tracked physics sim). Candidate for v2.

## 9a. Mirroring

Some datasets ship mirrored twins (SEED's `_M` files), others don't. Whether to use source mirrors is the
user's call; the tool offers both routes:
- Use source mirrors: they are ordinary input clips. Input filters (`--include` / `--exclude` glob,
  e.g. `--exclude '*_M.bvh'`) let the user drop or keep them.
- **Derive mirrors from our result**: `--mirror derive` writes `<name>_M.npz` next to each result by
  mirroring the *robot* trajectory (no second solve). `--mirror off` (default) / `derive`.
- Robot mirror = reflect the world across the robot's sagittal plane (y → −y in Z-up world):
  `root_pos.y` negated, root quaternion (w,x,y,z) → (w,−x,y,−z), left/right joints swapped, and per-joint
  sign flips. The swap/sign map is **derived from the USD joint axes** (not hand-written) and checked by
  an FK test: FK(mirror(q)) must equal mirror(FK(q)) for every link. Contact and force channels are
  swapped/reflected the same way.
- Mirroring is valid only for symmetric robots (both G1 variants are); the robot config declares it.
- The mirrored clip is **re-validated** and gets its own manifest row, with `mirror_of` pointing to the
  original. A rejected original yields a rejected mirror (or none with `--drop-rejected`).
- Plane convention is checked against SEED: mirroring the human of `X.bvh` should reproduce `X_M.bvh`,
  so derived and source mirrors are directly comparable.

## 10. Output formats

**NPZ** (written natively from Swift: zip of `.npy`, little-endian, float32 unless noted):

| key            | shape / type   | meaning                                        |
|----------------|----------------|------------------------------------------------|
| `fps`          | scalar f32     | output frame rate (default 50)                 |
| `robot`        | str            | `g1_23dof` / `g1_29dof`                        |
| `joint_names`  | [N] str        | canonical (Unitree SDK) order                  |
| `root_pos`     | [T,3]          | pelvis position, world, metres, Z-up          |
| `root_quat`    | [T,4]          | pelvis orientation, **wxyz**                  |
| `dof_pos`      | [T,N]          | joint positions, rad                          |
| `dof_vel`      | [T,N]          | joint velocities, rad/s                        |
| `root_lin_vel` / `root_ang_vel` | [T,3] | world frame                            |
| `body_pos` / `body_quat` | [T,B,3/4] | link poses (optional, `--with-bodies`)     |
| `ground_height`| scalar f32     | estimated ground height before the clip shift  |
| `contact_groups` | [G] str      | names of human contact groups (§5a)            |
| `contact`      | [T,G] u8       | per-group ground contact mask                  |
| `env_contact`  | [T,G] u8       | per-group environment-contact flags            |
| `tau`          | [T,N]          | joint torques, N·m (validator dynamics stage, floating-base RNEA) |
| `link_names`, `contact_force` | [L] str, [T,L,3] | net ground force per robot link, N (contact-force estimate) |
| `dynamics_residual` | [T]       | unexplained wrench per frame, units of m·g (§9) |
| `zmp`          | [T,3]          | ZMP of the motion on the ground; NaN while the ground carries < m·g / 2 |
| `status`       | str            | `accepted` / `rejected`                        |
| `reject_reasons` | [R] str      | empty if accepted                              |
| `metrics_json` | str            | full metric dict                               |
| `source`, `config_hash`, `version` | str | provenance                           |
| `mirror_of`    | str            | original clip if this is a derived mirror, else empty |

**Streaming** — `RobotFrame` (output) and `HumanFrame` (input), defined once in a schema file and
generated for both Swift (struct) and Python (`@dataclass`):

```python
@dataclass
class RobotFrame:
    stream_id: int        # batch slot / robot id
    seq: int
    t_source: float       # timestamp of the human frame this solves (s)
    t_emit: float         # when it was emitted (s)
    robot: str
    root_pos: np.ndarray  # (3,)
    root_quat: np.ndarray # (4,) wxyz
    dof_pos: np.ndarray   # (N,)
    dof_vel: np.ndarray   # (N,)
    contact: np.ndarray   # (G,) bool, per contact group (§5a)
    ground_height: float
    flags: int            # bitmask: clamped, infeasible, held, contact_uncertain ...

@dataclass
class HumanFrame:
    stream_id: int
    seq: int
    t_source: float
    skeleton: str          # "soma77" | "smplx"
    root_pos: np.ndarray   # (3,)
    joint_quat: np.ndarray # (J,4) local rotations, wxyz
    betas: np.ndarray | None  # SMPL-X shape, sent on first frame / on change
```
Wire: fixed little-endian binary layout with header (magic, version, type, length) over UDP
(low latency) or WebSocket (viewer/clients). Python client package: `retargeter_client`.

*As built (M6):* transport is **ZeroMQ** (libzmq 4.3.5, linked statically into `native/net`; pyzmq on the client):
the server binds a ROUTER socket (default `tcp://127.0.0.1:5560`), each client connects one DEALER socket and
sends `StreamStart` (skeleton: joint names, parents, rest offsets; fps) → `StreamInfo` (robot, dof names,
contact groups, lag), then `HumanFrame`s, each answered with a `RobotFrame` on the same socket, and `StreamEnd`.
Errors (unknown stream, malformed or undecodable message, no free slot) come back as `StreamInfo` with status 1;
the server keeps running. Messages are defined once in `Configs/schema/stream.json`; `tools/gen_stream_codec.py`
generates `Sources/MotionIO/StreamMessages.swift` and `python/retargeter_client/messages.py`. Arrays and strings
carry u32 counts; decoders reject bad magic / version, truncation, forged counts and trailing bytes.
`HumanFrame` carries every joint's local translation (`joint_pos`, J × 3) instead of a root position: SOMA BVH
moves both the virtual Root and Hips. All poses are in the canonical frame (m, Z-up, +X forward, wxyz); the
client's BVH reader applies the same conversion as the server's. Times are `CLOCK_MONOTONIC` seconds
(Python `time.monotonic()`), so on one machine `t_receive − t_source` is the transport delay. `RobotFrame.flags`:
1 velocity clamped, 2 cold start, 4 unsupported, 8 penetration, 16 environment support, 32 flight, 64 frames dropped.

## 11. Viewer

- Swift HTTP server inside the `retarget` binary (`retarget view`). Since the build is xmake (no SwiftPM
  package resolution), it is a small POSIX-socket server (`Sources/Viewer/HTTPServer.swift`, localhost
  only); WebSocket streaming for live teleop will be added to it in milestone 6.
- Front end: plain ES modules + vendored three.js — **no Node/npm build step** (Node isn't installed
  and isn't needed).
- Modes: **robot only / human only / both** (overlay, or side by side with offset).
- Timeline: play/pause, scrub, speed, frame step, loop; jump to reject ranges.
- Overlays: target markers + error lines, contact flags, support polygon + ZMP, CoM, torque heatmap on links.
- Panels: joint angle / velocity / torque plots vs limits; metrics table; accept/reject reasons.
- Sources: npz results, raw BVH / SMPL-X, the manifest (browse a batch, filter by status), and
  **live** teleop streams.
- Robot meshes: one-time export of `geometries.usd` → glTF (Python `usd-core` script in `tools/`).
- *Live (M6):* `retarget teleop --viewer-port N` serves `Web/live.html`; the server pushes, per open stream and
  at most 60 Hz, the robot's link transforms and the human joint positions as binary WebSocket frames (`/ws`;
  handshake and framing in `Sources/Viewer/WebSocket.swift`), plus a JSON message when streams open or close.
  One robot per stream side by side, with its human skeleton beside it; a status line shows the server delay
  and flags. A stalled browser is dropped (20 ms send timeout); it cannot hold up the solver.

## 12. Repository layout

```
retargeter/
  Package.swift
  Sources/
    CSlangBridge/   C++ → C API over Slang + slang-rhi
    Compute/        ComputeDevice, buffers, kernel cache, CUDA graphs
    Core/           math, RobotModel (USD parse), HumanSkeleton, Swift reference FK
    Dynamics/       floating-base RNEA, trajectory derivatives, contact-force estimate (NNLS)
    MotionIO/       BVH, SMPL-X npz, NPY/NPZ writer, stream codec
    Retarget/       Mapping, targets, RealtimeSolver, OfflineSolver, banded solver
    Validate/       metrics, reject rules, manifest
    Net/            ZeroMQ sockets (over native/net, module CRTNet)
    Viewer/         HTTP/WebSocket server, live viewer hub
    retarget/       CLI: offline | realtime | teleop | validate | mirror | view
  Kernels/          *.slang
  Web/              viewer (ES modules, vendored three.js)
  Configs/          robots/, mappings/, reject/, schema/ (wire messages)
  assets/           (existing) robots/, body_models/
  tools/            usd_to_gltf.py, gen_stream_codec.py (wire codecs)
  python/retargeter_client/
  Tests/
```

## 13. Milestones

0. **Toolchain spike** — *done 2026-10-23.* xmake builds Slang 2026.17.1 + slang-rhi (CUDA + CPU), the
   `rtcompute` bridge and the Swift `Compute` module. `Kernels/spike/chain_fk.slang` (FK + `fwd_diff` Jacobian
   + `bwd_diff` gradient) matches a double-precision Swift reference to < 1e-6 on both backends
   (`Tests/ComputeTests`). RTX 3090: 80 µs per dispatch for 4096 chains; CUDA kernel load 0.9 s cold,
   0.04 s from the on-disk cache. Findings: slang-rhi's CPU backend is ~500× slower (45 ms, appears
   single-threaded) and its kernels are not persisted by the shader cache (2.3 s load each run) — fine for
   parity tests, not for a CPU production path.
1. **Models** — *done 2026-10-23.*
   - USDA reader + `RobotUSDLoader`: G1-29 (32 links, 29 DoF, 33.3 kg) and G1-23 (28 links, 23 DoF,
     32.2 kg); joint order = Unitree SDK order; joint-frame FK matches the authored xforms to 4e-8.
   - `BVH` loader (SOMA uniform: 734 frames parsed in 14 ms); rest skeleton equals the SEED base rig.
   - Slang `robot_fk`, `skeleton_fk`, `lbs` kernels match the Swift double reference to < 1e-6 on CUDA and
     CPU (`Tests/KinematicsTests`). GPU LBS: 734 frames × 4,505 vertices in 46 ms.
   - `tools/bake_soma_surface.py` (SOMA-X + SEED shape) → `assets/body_models/soma/surface_lo.npz`;
     our LBS reproduces SOMA-X vertices to 5e-7 m; BVH-driven surface within 0.5 cm on the feet.
   - `tools/export_robot_meshes.py` → `build/viewer/robots/*.glb`; `retarget view` serves the viewer
     (robot / human / both, overlay or side by side, body + skeleton, demo robot motions).
1b. **Ground + contacts** — *done 2026-10-23.* Offline + causal (`Tests/ContactTests`, 8 behaviour clips:
   walk, kneel, crawl, push-up, lying, jump, sit on chair, stairs — all classified as expected; causal
   agrees with offline on 93–100 % of contact states, ~0.05 ms per frame). Viewer: contact-coloured body
   (green ground contact, red environment contact), ground plane, ZMP/CoM markers, contact tracks,
   offline/causal switch. The compute bridge now makes the CUDA context current on every call, so devices
   work from any thread.
2. **Kinematic retarget** — *done 2026-10-23.*
   - Mapping configs `Configs/mappings/soma77_g1_{29,23}dof.json`: 18 targets (pelvis, torso, hips, knees,
     ankles, toes, shoulders, elbows, wrists + hand orientation), per-chain scales with a uniformly scaled
     root (pelvis-height ratio 0.785), rotation offsets from the SOMA T-pose against an explicit robot
     reference pose (arms out, elbows straight — the G1 elbow is bent 90° at q = 0; a solved T-pose
     landed in a twisted arm configuration and is only the fallback).
   - Solver (`Kernels/retarget_ik_*.slang`): warm-started Levenberg-Marquardt per frame (8 iterations),
     analytic geometric Jacobian, smoothness / posture / soft-limit regularisers, hard limit clamp. GPU
     variant: one 32-thread group per stream, cooperating through groupshared memory (~1.2 ms per frame per
     stream on the RTX 3090, streams in parallel); CPU variant: one thread per stream (Slang's CPU target has
     no group barriers). CUDA vs CPU agree to 2e-6 rad; batched == individual exactly.
   - Ground: causal tracker per clip (what a live stream would see); goals are built after the ground shift.
   - Quality on SOMA clips (G1-29): mean target error 29–50 mm, p95 56–135 mm, 0 limit violations;
     robot-generated motion is recovered to 0.7 mm mean.
   - `NPZWriter` + `RobotMotionExport` (50 fps default, keys as §10 plus contact masks, ground height,
     tracking cost, human flags). `retarget realtime <inputs> --robot … --out …`: batched, resumable,
     per-clip failure isolation, `manifest.jsonl`. 32 clips / 88k frames in 18 s.
   - Viewer: "Retargeted (realtime)" robot motion next to the human clip.
   - Compute bridge: kernels compile on a dedicated 64 MB-stack thread (Slang crashed on HTTP worker
     threads), so devices and kernel loading work from any thread.
3. **Validator** — *done 2026-10-23.* `Sources/Validate/`, rules in `Configs/reject/g1.json`.
   - Human pre-filter (on the offline contact analysis): env contact / support, unexplained unsupported runs,
     penetration, ground tilt (warn), windowed root jerk (≈3 Hz smoothing, 30 Hz, 4 s windows), length,
     teleports, non-finite values.
   - Robot metrics: tracking (ours only), limit and velocity violations (velocity limits: assumed 20 rad/s
     in `Configs/robots/g1.json` until confirmed), joint jumps, accel/jerk p99, root teleports, foot skate /
     float against the human foot contacts (the contact-carrying sole point is followed across frames),
     ground penetration of every link surface point (`mesh_points.npz`, exported with the meshes),
     self-collision (surface points inside the other link's median-radius capsule; pairs > 2 joints apart
     and clear in the default pose), start tilt, start/end static stability (CoM over every
     ground-touching surface point).
   - `retarget realtime`: human stage before the solve, robot stage on the exported 50 fps trajectory;
     `accepted/` / `rejected/` folders, `status`, `reject_reasons`, `warnings`, `metrics_json` in the npz,
     full report per manifest row; `--no-auto-reject`, `--drop-rejected` (human rejects then skip the solve),
     `--reject-config`. `retarget validate` checks any npz / soma-retargeter CSV against its source clip.
   - Comparison with NVIDIA soma-retargeter on 10 clips: `docs/baseline-comparison.md`. Neither passes the
     strict robot rules; NVIDIA is ahead on self-collision (2.0 vs 3.7 cm), foot skate and penetration —
     our hands sink into the hip housings because per-chain scaling narrows the shoulders. Ours is
     smoother and ~9× faster. Drives the M4 priorities: self-collision and ground terms, foot locking.
   - Solver fix found by the validator: the first frame of a cold stream now converges with 50 iterations
     and no smoothness pull (it caused one-frame joint jumps).
4. **Offline** — *done 2026-10-23.* `retarget offline` (same options as `realtime`), viewer "Retargeted (offline)".
   - Objective over the whole clip: tracking (goals on the non-causal offline ground), velocity and
     acceleration smoothness to both neighbours, stance-foot locking (each sole corner held at its stance
     run's mean ground position while the human heel / toe is down), ground contact equality for robot links
     mapped from human body-part contacts (`contacts` in the mapping), a one-sided ground barrier on every
     link's lowest surface point, capsule self-collision clearance (enclosing capsules, 1 cm margin), posture,
     soft limits + clamp.
   - **Deviation from the plan:** instead of a block-banded Cholesky over time, the objective is minimised by
     coloured block Gauss-Seidel: frames coloured t mod 3 (neighbours t±1, t±2 never share a colour), one
     cooperative LM step per frame per sweep with neighbours fixed, 150 sweeps, constraint weights ramped up over
     the first half (continuation). It parallelises over every frame of every clip and reuses the realtime
     kernel's structure. Warm start: the realtime solution.
   - Results (10 comparison clips, `docs/baseline-comparison.md`): 5/10 accepted vs 0 for realtime and NVIDIA;
     ground penetration 1.6 cm, self-collision 0.4 cm (means of per-clip max). 23 s for 16k frames.
   - Tuning lesson: strong one-sided terms with one GS step per sweep chatter (one-frame shoulder-yaw flips);
     the acceleration weight (60) has to grow with the constraint weights, with more sweeps to compensate.
   - CUDA only (the CPU backend lacks group barriers and the serial variant was not written).
5. **Dynamics** — *done 2026-10-26.*
   - `Sources/Dynamics` (Swift, double): floating-base RNEA in the world frame with external point forces,
     link motion, momentum, kinetic energy; trajectory derivatives (central differences, Gaussian-smoothed
     velocities); contact-force estimate by non-negative least squares (Lawson-Hanson, warm-started from the
     previous frame) over friction-pyramid edges. Tests (`Tests/DynamicsTests`): static torques = ∂V/∂q, base
     wrench = momentum rate − gravity, power balance τ·q̇ + base wrench · base twist = dE/dt, Jacobian
     transpose / virtual work, contact forces of a standing robot, NNLS against brute force.
   - Validator dynamics stage (`Sources/Validate/DynamicsCheck.swift`): contacts are robot surface points on
     the ground (or near it and slow), extended ±2σ in time because velocity smoothing spreads impacts; per
     frame the contact forces, joint torques and the unexplained wrench. Metrics and rules
     (`Configs/reject/g1.json`), each held for a window so single noisy frames do not count:
     `dyn_residual_max` (unexplained wrench held 0.1 s, |f|/mg ⊕ |n|/(mg·0.1 m)) ≤ 0.25, `zmp_outside_max`
     (held 0.1 s) ≤ 5 cm, `joint_acc_ratio_max` (|q̈| / effort-limited acceleration, held 50 ms) ≤ 1;
     warning on the residual p95. Joint torques (`torque_ratio_max`, |τ| / effort held 50 ms) are recorded but
     not judged (a reject rule, then a warning, until 2026-10-26, see §14). Torques, contact forces, residual and ZMP are written into the npz. Effort limits are the
     USD assets' `mjc:actuatorfrcrange` (G1-29: hips 88/139, knees 139, ankles 50, waist 88/50, shoulders and
     elbows 25, wrists 25/5 N·m). `retarget validate` runs the robot stage of a batch of clips in parallel.
   - Offline solver (`Kernels/offline_ik_group.slang`): root acceleration smoothing (the M4 solver smoothed
     joint but not root accelerations — the dynamics metrics exposed 30 m/s² median root jitter); centroidal
     rows for the frames t − K, t, t + K (K ≈ 25 ms: frames are now coloured mod 2K + 1) with point-mass
     links and contact forces re-solved in the kernel at the stance anchors / mapped-contact patches; a
     per-step trust region. (Torque rows for unloaded joints were tried and removed: no effect on any judged
     metric.) Tuned: dynamics 3, ground 200, contact 100, 5 warm-started contact-force passes.
   - Self-collision near the hands (found in the viewer, 2026-10-26): clasped or rubbing hands made the offline
     hands jump (hand accelerations up to 180 m/s², realtime ≤ 22). At robot scale the hands overlap, and the 9
     separate link pairs between the two hands pushed them apart in different directions from frame to frame. Now
     one hand-to-hand row (deepest overlap of those pairs, hand capsules at the 90th radius percentile, pushed
     along the centre-to-centre direction). Peaks 23–64 m/s² at that point; with the shoulder anchors below, dish
     washing went to the realtime level. (Spreading warm-start overlaps to neighbouring frames, `collision_spread`,
     helped at first and was removed once the arms were anchored.)
   - Arms anchored at the shoulders (2026-10-26, the main fix for hands colliding with the body): per-chain
     scaling from the pelvis uses the spine-dominated chain ratio (~0.59) for the arms too, which put the G1's
     shoulder goals 9.9 cm from the pelvis axis (robot: 14.1 cm) and the hands into the hip housings (16.4 cm
     half-width; the human's hips are narrower than the shoulders, the G1's hip surface is wider). Now
     `scale.anchors` (LeftArm, RightArm): the shoulder goal is moved sideways (in the chest frame) to the robot's
     width, elbows and hands hang from it scaled by the arm's own length ratio. Offline relaxed walk: 120 Hz hand
     jitter frames 419 → 24, kick trash 31 → 0, dish washing 82 → 0; comparison clips 6/10 accepted, self-collision
     0 on every clip, tracking 5.5 → 5.1 cm. With it the time-spread clearance (tried earlier) only hurt and was
     removed; hand smoothing, the smooth barrier and the hand-to-hand row stay.
   - Shaking near the body (viewer, 2026-10-26: relaxed walk, kick trash, dish washing). Measured at the solver's
     120 Hz: the whole offline body jittered (pelvis second difference p99 3.5 mm, realtime 0.7) and the hands more.
     Causes and fixes: (1) the stance lock switched each sole corner on at full weight and pulled it to the ground
     while the person's heel was still up — its vertical pull now ramps in / out over 50 ms (`lock_ramp`; the
     horizontal lock stays full, ramping it let the feet slide); (2) the collision push switched its full
     stiffness on at contact — now a smooth barrier (quadratic ramp over `collision_band` = 3 cm before the
     margin); (3) light Cartesian acceleration rows for the hands (`hand_acceleration` = 100; stronger ones fight
     the centroidal term through the root). Result on the relaxed walk: pelvis 0.97 mm; hand jitter max
     22 → 8 mm, frames with > 2 mm second difference 629 → ~420; kick trash 97 → 31; dish washing 172 → 82; no
     self-collision. Remaining: in relaxed walking the person's arm swings through where the robot's hip is, and
     the robot hand detours around it quickly (smooth but fast). Tried and dropped: pushing hands sideways off the
     legs (helps walking, hurts hands in front of the thighs), shifting hand goals out of overlaps (no effect),
     arm-only hand smoothing (worse), strong hand smoothing / collision (breaks the dynamics). Cost of the ramp:
     jump unexplained wrench 0.14 → 0.17 m·g (rule 0.25).
   - Realtime solver: the same centroidal rows, causal, on the GPU variant — measured worse at lag 0 and off
     by default (§7).
   - Results (10 comparison clips, `docs/baseline-comparison.md`): offline mean unexplained wrench 0.06 m·g
     (realtime 0.16, NVIDIA 0.22), ZMP excursion 2 cm (7.5, 14), ground penetration 0.5 cm (7.1, 6.3);
     torque ratio 1.76 (1.17, 1.00). With the feasibility rules as finally set (torque not judged, joint
     acceleration ≤ the effort-limited value) offline accepts 5/10, realtime and NVIDIA 0/10;
     jump (ankle 1.5× its torque limit on landing) and push-up (5 N·m wrists 2.1×) pass. Worst joint
     acceleration ratio over the clips: 0.70 (offline body check, shoulder pitch). Offline run: 22.5 s for the 16k
     frames (prepare 4.2 s, realtime warm start 6 s, sweeps 11.4 s, validation 0.3 s; M4: 23 s). It was 62 s
     before the speed-up: the host recomputed stance anchors and contact patches for every frame on every sweep
     (34 s) — now parallel over frames and refreshed every 5 sweeps (`anchor_interval`) — and the in-kernel
     contact-force solve now warm-starts from the previous sweep (5 coordinate-descent passes instead of 30).
   - Limitations: link rotational inertia is ignored in the solver (the validator has it). Friction μ = 0.7 is
     assumed.
6. **Teleop** — fixed-lag realtime, UDP/WebSocket streams, Python client, live viewer.
   - *Phase A done 2026-10-27 (lag 0):* `retarget teleop` (ZeroMQ ROUTER server, §10), `LiveRetargeter` (§7),
     `python/retargeter_client` (client, BVH reader, `python -m retargeter_client clip.bvh …` real-time replay with
     latency report), live viewer (§11). Streamed output equals the batch realtime solve bit for bit in process
     (`RetargetTests`) and to float32 input rounding over the wire (≤ 1e-6 rad on a 10 s walk). End-to-end 1.9 ms
     median / 3.1 ms p95 for one stream at 120 Hz, ≤ 4.6 ms p95 for four; no dropped frames in a soak run with
     streams opening and closing. Tests: wire codec (`CoreTests`, incl. bytes from the Python encoder) and
     `python/tests/test_teleop.py` (server + client end to end: ordering, latency, one cold start, error replies).
   - Phase B (2026-10-27): the fixed-lag window (§7 as built), solved at the 50 Hz policy rate, with a reference
     prior, lagged contact flags and a light dynamics term: 6/8 comparison clips accepted at 40 ms / 5 sweeps (lag 0:
     0/8, offline 7/8); contacts, collisions, smoothness and ZMP at offline level, tracking ~1.2 cm behind, dynamics
     residual twice offline's. Live 58 ms median / 67 ms p95 end-to-end. Open: tracking, the dynamics weight's small
     margin (2x), the stairs clip.
7. **SMPL-X** input path and mappings.

## 14. Decisions log

- NPZ output fps: **50** by default (`--fps`).
- Bad clips: **rejected whole**; no segment trimming in v1.
- Simulation-tracking validation: **not in v1**.
- Mirrored clips: user chooses whether to use source mirrors; tool can **derive mirrors** from its own results (§9a).
- Ground is **auto-estimated from human body-surface vertices**; contacts are detected for **any body part** (§5a).
- Ground contacts of any mapped body part are robot contacts; environment support → offline reject (v1).
- Offline dynamics term keeps the contact-force solve (2026-10-26): a force-free CoM-support variant (no pull,
  friction on the total force, ZMP in the support polygon) diverged on 5 of 8 comparison clips
  (`docs/baseline-comparison.md`); it stays as the experimental `offline.dynamics_model = "force_free"`.
- Validator corrections after visual checks (2026-10-26):
  - Foot skate is judged only where the person's foot is planted (touching heel / toe slower than 10 cm/s,
    `skate_planted_speed`) and the robot's sole is on the ground (< 2 cm); the person's own baseline uses the same
    frames. A dragged foot (crawling) may drag on the robot too; a robot foot in the air is foot_float's business.
  - The ZMP rule rejects only excursions held for 0.3 s (`dynamics_zmp_window`, was 0.1 s): at heel strike and
    toe-off the smoothed ZMP leads the instantaneous contact set by up to ~5 cm for 40-100 ms on visibly stable
    walking; a sustained lean still fails.
- Offline stance lock: a sole corner is locked during a stance run only if it stays within 4 cm of the foot's lowest
  corner (`lock_height`): a robot on its toes while the person's whole foot touches (crawling) had its raised heel
  corners pinned to the ground, dragging the toe.
- Hands kept apart along the person's hand-to-hand direction (2026-10-26): the hand-to-hand row pushes from the
  right hand's goal to the left's, so the robot separates its hands as the person's are and switches only when
  theirs do (with the robot's own hand centres it flipped one hand above / below the other). Hand smoothing 300
  (stronger fights the dynamics through the root). Pairs between the two arms (shoulder, upper-arm, forearm and
  hand links) likewise push along the closest-point direction of the person's scaled arm segments the links stand
  for (shoulder point, shoulder -> elbow, elbow -> hand, hand point), not the robot's own: a hand caught between the
  chest and the other arm (230127 body check A155, ~4 s) escaped below / above it (peak hand acceleration 70 -> 17
  m/s²). An arm against the pelvis or a leg pushes along the person's direction (hips point; hip point, hip ->
  knee, knee -> ankle) blended 2:1 with the robot's own: a hand sliding down past the wider robot hip swung from
  its front to its side within a few frames (A155 14.8 s, 63 -> 45 m/s²); the person's direction alone gave 25
  but left a hand up to 1.4 cm inside the hip on 230322 A296, and 4:1 did so on A155. Remaining peaks (~35 m/s²
  at 13.6 s, hands landing on the hips) match the person's motion (realtime 29-33 there).
- Non-foot ground contacts ramp in / out (2026-10-26): a link held on the ground (hand, knee, ...) gets its pull
  over `contact_ramp` (0.1 s) at both ends of the contact run; at full weight a hand whose person's fingertips touch
  the floor while the scaled hand target is ~7 cm higher dropped and jumped back up in a few frames (A155 20.6 /
  20.9 s, 49 -> 22 m/s²). Below the ground the row keeps at least the ground weight (a lightly weighted contact let
  knees and hands sink 5-7 cm on crawl / push-up / stand-up).
- Self-collision allows touch (2026-10-26): the person's hands touch their body (230127 body check), so the solver
  keeps capsule axes apart by the validator's own geometry — max(enclosing radius of one + median radius of the
  other, and the reverse) — with no margin, instead of two enclosing capsules plus 1 cm (which kept a hand ~5 cm
  off the boxy G1 torso); the validator's 1 cm penetration tolerance stays.
- Dynamic feasibility = CoM support + joint velocity + joint acceleration (2026-10-26). Rejects: unexplained
  wrench ≤ 0.25 m·g (held 0.1 s) and ZMP ≤ 5 cm outside the support (held 0.3 s), joint velocity ≤ limit, joint
  acceleration ≤ the effort-limited acceleration (held 50 ms). Joint torques are recorded, not judged (no
  reject, no warning). The realtime (lag 0) solver has no dynamics term.
- Joint torques (2026-10-26): computed only in the validator — RNEA is needed there anyway for the CoM-support
  check, and torques come with the contact forces at no measurable cost; exported as `tau`, not judged. The
  offline solver has no torque term.
- Feasibility levels (survey §6): kinematic validity, contact consistency, source balance and dynamic feasibility
  (contact wrench cone) are checked; simulation trackability is v2.
- Joint acceleration limits are derived from the model per robot: effort / (inertia beyond the joint about its
  axis + armature) in the default pose, no gravity load (G1-29: hip pitch 95, waist pitch 80 / roll 71, shoulder
  pitch 178, elbow 562, knee 1105, ankles ~4000 rad/s²); `acceleration_limits` in `Configs/robots/g1.json`
  overrides single joints.
- Throughput (2026-10-26). On the 10 comparison clips (16k frames, RTX 3090) realtime goes from 10.9 s to
  3.5 s and offline from 23.0 s to 5.1 s, with every status and validator metric unchanged (outputs equal to
  float rounding; offline joints within 0.002 rad). What changed:
  - Realtime kernel (6.1 → 1.8 s): forward kinematics level-parallel over the kinematic tree instead of serial
    on one lane; the normal equations summed only over the row blocks that both columns share (per-column target
    masks, zero-weight parts skipped; no clearing of the Jacobian); a sparse Cholesky in minimum-degree order
    (limbs first, root last: 294 factor entries below the diagonal instead of 595), its trailing updates spread over the lanes and
    fused with the forward substitution; the dynamics rows skipped at weight 0. Wave intrinsics are avoided: one
    `WaveReadLaneAt` made Slang wrap every branch of the kernel in `__ballot_sync` (3.5x slower).
  - Offline kernel (sweeps 12.0 → 3.2 s): same forward kinematics and Cholesky work (dense, as the dynamics rows
    couple every column), the normal equations over per-column row bitsets; the lowest point of a link is
    searched only if its bounding sphere reaches the ground margin or it is in contact (all lanes scanning);
    collision pairs whose bounding spheres are farther apart than the band are skipped; the contact-force solve
    works on the active points only, with their moment arms computed once; the tracking-target rows are not
    stored but evaluated in closed form, which brings the group memory from 47 to 31 KB (3 groups per SM instead
    of 2), and each frame uses two warps.
  - Offline anchors: the sweep before a refresh writes the sole points and contact patches of its result, so
    the host no longer downloads the trajectories and runs FK per frame (2.2 s of idle device per run).
  - Offline warm start in segments (§8; 1.6 → 0.2 s).
  - Preparation on the CPU in parallel (4.0 → 0.8 s), and the three stages overlapped across batches.
  Realtime iterations stay at 8 (decided 2026-10-27, quality first): 4 would halve the solve at up to 0.55 deg joint difference.
  At scale (2026-10-27; 128 random SOMA clips, 116k frames, batch 32): 22 s wall, about 5,200 frames/s; prepare
  (3.5 s) and validate + export (1.7 s) stay hidden behind the solve (19 s). The anchor refresh was the one stage
  the device waited on (2.2 s); computing the anchors in place on the batch arrays, with no per-clip and
  per-frame copies, brought it to 0.9 s (bitwise the same anchors), most of it the transfers and the first
  refresh's host FK. Batch 64 is ~1.5 s slower on 128 clips (two batches overlap less).
  Offline runs are bit-reproducible, and independent of the batch size: the collision pair rows used to take
  their slot by an atomic counter, so their order in the normal equations, and which rows were dropped when the
  slots were full, followed thread timing (joints differed by up to ~0.0015 rad from run to run). They now take
  rows in pair order (per round of 64 pairs, the count of active pairs before each), at no measurable cost.

- Teleop transport: **ZeroMQ** (2026-10-27, user choice) instead of raw UDP / WebSocket; one ROUTER/DEALER socket
  pair carries control and frames both ways (no PUB/SUB slow-joiner loss, replies routed per client). The browser
  viewer stays on WebSocket. M6 is staged: lag-0 streaming first, the fixed-lag window second.

- Fixed-lag window (2026-10-27): built on the offline kernel; emitted frames are frozen (never revised), the window
  is the not-yet-emitted frames; no dynamics term in the window (see §7 as built). Solved at the policy rate,
  50 Hz (user: the policy runs at 50 Hz, so 120 Hz output is not needed; balance latency and quality): default
  lag 40 ms. Second round: an absolute prior to the realtime solve (drift was integrated through each new frame's
  starting point), contact flags decided a few frames late with the offline test, dynamics at weight 1; 5 sweeps.

- Validator balanced (2026-10-27; user: many visibly fine clips were rejected). On a 128-clip offline sample
  84 → 109 accepted; 5 of the user's 6 example clips now pass:
  - Foot skate is measured per planted part (heel / toe half of each sole) once it has been in contact for 40 ms
    (`skate_settle`): the slide is the smallest centroid displacement of a planted part's grounded corners, so a
    heel-to-toe roll is no longer counted as a slide.
  - Root jerk (window max): warn above 100, reject above 200 m/s³ (600 clips: p90 50, p99 94, max 122); was a
    reject at PHUMA's 50.
  - Dynamics residual split: `dyn_residual_max` (≤ 0.25) on supported frames only, `dyn_flight_residual_max`
    (≤ 0.40) on frames with no contact, where small smoothing errors at take-off / landing are amplified.
  - Joint acceleration is judged against the pose-dependent effort-limited acceleration (inertia at that frame's
    pose), not the straight-leg worst case; `self_collision_max` 0.015 (the capsules are only good to ~1 cm).
  - Still rejected: ib_dodge_up_R_001__A437, `waist_pitch` at 1.36× its effort-limited acceleration with the arms
    raised (inverse-dynamics torque ~20 % over the limit) — a real overload, left as a reject.
- Hand rotation targets weigh the twist about the forearm at 0.25 (`"twist"`, `"twist_axis"` per target; residual
  and Jacobian rows scaled about the link axis). Captured hand twist is the least reliable rotation in the data
  (crowd_cheer_001__A029: −0.7 → 2.3 rad at up to 59 rad/s) and drove the wrist roll past its velocity limit.
  Neutral on the comparison clips (status unchanged, tracking within 0.03°); large steps on crowd_cheer 38 → 24
  offline, 31 → 28 realtime. `solver_limits` (per-joint solver-only range, mapping JSON) added but unused by default.

- Arm branch switch (2026-10-28; user's pick after the shoulder yaw limit failed). The realtime GPU kernel
  (`retarget_ik_group.slang`, `branchSwitch`; mapping `solver.branch_*`): every 50 ms an arm more than 5 cm off
  is solved again from its joints at 0 (20 iterations, no smoothness pull). A branch that halves the arm's
  tracking cost is kept as an alternative and solved warm every frame next to the current one. The output arm
  moves toward a branch once it has tracked 2× better for 0.2 s (`branch_dwell`), with a smoothstep over 0.3 s
  (`branch_blend`). Reaching it swaps the two; the other is dropped after 1 s (`branch_keep`) or once they merge.
  Offline and the fixed-lag window inherit it through their realtime warm start / reference.
  - jump_right_003__A030: peak shoulder speed 55 → 14 rad/s at 50 Hz output (93 → 12 at 120 Hz), realtime
    mean cost 0.42 → 0.17; offline now accepted; realtime and lag have no joint jumps or velocity violations
    (left: self-collision in realtime, in-air dynamics residual).
  - Instant switching (the removed re-seed) moved joints up to 3 rad in a frame. Dropping the old branch at once
    switched crowd_cheer into one that went bad a second later. Switching on every cost crossing (no dwell) swung
    its shoulder 3 rad back and forth: the arm pumps overhead and each branch wins in turn.
  - Comparison clips: bit-identical offline, realtime and lag (no switch fires). Teleop p95 unchanged (1.7 ms lag 0,
    68 ms lag 40 ms). The CPU kernel (`retarget_ik.slang`) has no switch.

- Contacts and thresholds (2026-10-28; user: contacts wrong on step-downs / sitting, rejects a hair over the
  bound, rethink the thresholds). Calibrated on the 128-clip sample: 111 → 123 accepted; comparison clips 6 → 7.
  - Environment contact: a static upper body explained the ZMP on step-downs (torso, head, pelvis, knee on
    come_down_50cm_box) because a part above the CoM always "contains" it. Now a part must be free below and
    still in 3D (burpee knees / thighs swinging up counted). Box clips keep only env support (the box); sitting
    keeps the pelvis and armrest elbows.
  - Support hull includes parts on the ground but too fast for a contact: a sliding foot (dance slide, sideways jog
    push-off) left the ZMP "unsupported".
  - Penetration against the local floor: a not-quite-level capture floor drifted 3-5 cm over a sideways jog.
    A whole-clip plane fit was tried and rejected (tilt estimates unstable: 2.1° → 0.06°, 1.7° → 4.0° between
    fits). Limbs / trunk 6 → 8 cm: kneeling knees and thighs sink 6-7 cm under body weight (no flesh in LBS).
  - Rules round the value to two significant digits of the bound (0.25: 0.01 steps; ≤ 5 % over passes) and print
    the digits that decided ("0.016 > 0.015", not "0.015 > 0.015"); a rule's `precision` overrides.
  - Joint velocity limits from Unitree's g1_29dof.urdf (hips / waist yaw 32, knees 20, ankles / waist 30, arms 37,
    wrist pitch / yaw 22 rad/s; was 20 everywhere). Joint jump is now a step faster than 1.5× the joint's limit:
    the old 0.35 rad per output frame was 17.5 rad/s at 50 Hz, stricter than the velocity limit itself.
  - foot_skate_excess 0.05 → 0.08: turning jogs reach 0.05-0.08 from 20-60 ms slides at touchdown / toe-off
    (~5 mm each). Warnings recalibrated to the sample's top ~5 %: dyn_residual_p95 0.10 → 0.20 (warned on 30 %),
    foot_skate_p95 0.10 → 0.15.
  - Human ZMP from all body parts as point masses instead of the CoM alone (user: turning jogs such as
    neutral_dancecard_go_turn_jog_001__A536 flagged unsupported): clears that clip and the burpee's human
    rejects; chair, stairs and box still caught, no other human-side change on the sample.
  - Seated contacts (user: having_a_sit_relax_001__A005 labelled an arm but not the pelvis and thighs): parts on
    the same surface as a chosen one are labelled too, and labels cover the whole rest; chair clips now show pelvis,
    thighs and knees for the whole sit. The torso is labelled where the support search needs it (likely a backrest;
    the model cannot tell). No change to the sample's rejects.
  - Human unsupported judged in seconds (`unsupported_seconds`, total unexplained time): warn above 0, reject above
    0.5 s (user: brief unsupported frames are possible for a person and a robot alike; was `unsupported_frames` > 0).
    Only runs of >= 0.3 s with the ZMP > 6 cm outside count at all; leftovers in box / stairs / chair clips are
    0.03-0.53 s, none in the sample outside a box clip. `unsupported_frames` is still recorded.
  - Still rejected in the sample, judged real: box step-down, burpee (robot: residual, ZMP), jump_and_land_professional (landing over effort-limited acceleration), dance slide (ZMP and
    residual), sit cross-legged (tracking, self-collision).

## 15. Open questions

- Arm problems (2026-10-27; jump_right fixed 2026-10-28 by the branch switch, §14): jump_right_003__A030 falls into a shoulder gimbal branch (right
  shoulder roll −2.22, yaw 2.57, both at their limits, 1.4 – 3.4 s, then snaps at 3.58 s; elbow up to 14 cm off,
  while a cold solve from 1.9 s finds a 2 – 3 cm branch). Re-seeding from a cold solve fixed the trap but caused
  joint jumps up to 3 rad, so it was removed. crowd_cheer_001__A029 is a source hand-twist artefact, only partly
  mitigated by the twist weight. Candidates: a narrower solver range for shoulder yaw (±1.8 tried: realtime large
  steps 31 → 16 on crowd_cheer, 5 → 3 on jump_right, roll still trapped), a blended branch switch, a human-stage
  hand-twist-rate warning. Tried (user's pick): solver-only shoulder yaw ±1.8 on both arms — comparison clips
  bit-identical, but on jump_right the snap only moves from 3.6 s to 1.9 s (shoulder pitch ~57 rad/s realtime,
  roll still pinned at −2.22); lag joint-acceleration ratio 1.2 → 2.5. Not kept. Still open: crowd_cheer_001__A029
  (both arm branches sit at shoulder limits while the arm pumps overhead, plus the source hand twist).
- Dynamics thresholds are calibrated on 10 clips only; friction (0.7) and the G1 effort limits (taken from the
  USD assets' MuJoCo actuator ranges) should be checked against Unitree's specification, as should the derived
  acceleration limits (they ignore gravity load and use the straight-leg pose, the largest hip inertia).

- BONES-SEED `soma_shapes/` (uniform-skeleton shape params) is gated on Hugging Face — needed locally for
  the SOMA body-surface bake; otherwise we fall back to the SOMA-X template body or capsules.
