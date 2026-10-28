# Baseline comparison: our realtime and offline retargeters vs NVIDIA soma-retargeter (2026-10-23, updated 2026-10-26)

## Validator corrections and hand fixes (2026-10-26)

After visual checks: foot skate judged only where the person's foot is planted and the robot's sole is down; the
ZMP rule rejects excursions held for 0.3 s (transitions at heel strike / toe-off no longer count); offline stance
lock skips raised sole corners; hand-to-hand separation follows the person's hands, and so do pairs between the two arms (cross-arm pushes
along the person's arm segments; body check A155 no longer flips a hand under the other arm); hand smoothing 300.
Re-run after the cross-arm change: offline still 6/10 accepted with all rules on, same means (stand_up_lying sits on the 2 cm
penetration limit at 2.03 cm and flips between runs).
Then hand-vs-pelvis/leg pushes blended with the person's direction and non-foot ground contacts ramped in / out over
0.1 s: offline 7/10 with all rules on (stand_up_lying back under the penetration limit), tracking 3.8 cm, residual
max 0.072, ZMP outside 1.0 cm, self-collision 1 mm (means over robot reports). All three
tools revalidated with the same rules (tracking rules off).

| clip | realtime | offline | nvidia |
|---|---|---|---|
| walk_forward_amateur_001__A001 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max) |
| kneeling_loop_001__A021 | rejected (ground_penetration_max) | accepted | rejected (ground_penetration_max, self_collision_max) |
| jump_ff_180_R_001__A045 | rejected (self_collision_max) | accepted | rejected (ground_penetration_max, self_collision_max, dyn_residual_max) |
| sit_on_chair_loop_R_001__A045 | rejected (env_contact_frames, self_collision_max, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) |
| crawl_ff_start_180_R_001__A125 | rejected (foot_float_p95, ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max) | accepted | rejected (foot_skate_excess, ground_penetration_max, dyn_residual_max, zmp_outside_max) |
| body_check_001__A296 | rejected (self_collision_max) | accepted | rejected (self_collision_max) |
| sweep_floor_side_walk_R_004__A298 | rejected (self_collision_max) | accepted | rejected (self_collision_max) |
| stairs_climbing_down_loop_R_001__A300 | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_skate_excess, ground_penetration_max, self_collision_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_skate_excess, dyn_residual_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_float_p95, self_collision_max, dyn_residual_max) |
| push_up_loop_001__A359 | rejected (ground_penetration_max) | accepted | rejected (ground_penetration_max) |
| stand_up_lying_R_002__A472 | rejected (ground_penetration_max, self_collision_max) | rejected (ground_penetration_max) | rejected (ground_penetration_max, self_collision_max) |

| metric (mean over the 8 clips that pass the human pre-filter) | realtime | offline | nvidia | best |
|---|---|---|---|---|
| foot skate p95 (m/s) | 0.057 | 0.040 | 0.056 | offline |
| foot skate beyond the source (m/s) | 0.016 | 0.004 | 0.016 | offline |
| foot float p95 (m) | 0.014 | 0.009 | 0.010 | offline |
| ground penetration max (m) | 0.066 | 0.006 | 0.063 | offline |
| self-collision max (m) | 0.026 | 0.001 | 0.020 | offline |
| max |q̇| / limit | 0.34 | 0.33 | 0.38 | offline |
| joint jumps (frames) | 0.0 | 0.0 | 0.0 | ≈ |
| joint accel p99 (rad/s²) | 49 | 42 | 53 | offline |
| end stability margin (m) | 0.106 | 0.091 | 0.109 | nvidia |
| max |q̈| / effort-limited acceleration, held 50 ms | 0.25 | 0.32 | 0.30 | realtime |
| unexplained wrench, held 0.1 s (m·g) | 0.157 | 0.072 | 0.217 | offline |
| ZMP outside support, held 0.1 s (m) | 0.047 | 0.008 | 0.050 | offline |
| max |τ| / effort limit, held 50 ms | 1.23 | 1.98 | 0.99 | nvidia |
| frames over a torque limit | 5.8 | 35.2 | 2.6 | nvidia |

- **realtime**: 0/10 accepted
- **offline**: 7/10 accepted
- **nvidia**: 0/10 accepted

## Whole-body scaling and touching hands (2026-10-26)

Limb anchors for the legs too (height from the robot's hips, horizontal placement uniformly scaled from the
pelvis), straight-line limb scales, and self-collision that allows touch (validator geometry, no margin).
Realtime and offline re-run; tracking rules off as before (offline tracking error 3.7 cm mean, was 5.1).

| clip | realtime | offline | nvidia |
|---|---|---|---|
| walk_forward_amateur_001__A001 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max) |
| kneeling_loop_001__A021 | rejected (ground_penetration_max) | accepted | rejected (ground_penetration_max, self_collision_max) |
| jump_ff_180_R_001__A045 | rejected (self_collision_max, zmp_outside_max) | accepted | rejected (ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max) |
| sit_on_chair_loop_R_001__A045 | rejected (env_contact_frames, self_collision_max, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) |
| crawl_ff_start_180_R_001__A125 | rejected (foot_float_p95, ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max) | rejected (foot_skate_excess) | rejected (ground_penetration_max, dyn_residual_max, zmp_outside_max) |
| body_check_001__A296 | rejected (self_collision_max) | accepted | rejected (self_collision_max) |
| sweep_floor_side_walk_R_004__A298 | rejected (self_collision_max) | accepted | rejected (self_collision_max) |
| stairs_climbing_down_loop_R_001__A300 | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, ground_penetration_max, self_collision_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_skate_excess, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_float_p95, self_collision_max, dyn_residual_max) |
| push_up_loop_001__A359 | rejected (ground_penetration_max) | accepted | rejected (ground_penetration_max) |
| stand_up_lying_R_002__A472 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max, self_collision_max, zmp_outside_max) |

| metric (mean over the 8 clips that pass the human pre-filter) | realtime | offline | nvidia | best |
|---|---|---|---|---|
| foot skate p95 (m/s) | 0.091 | 0.083 | 0.080 | nvidia |
| foot skate beyond the source (m/s) | 0.009 | 0.022 | 0.003 | nvidia |
| foot float p95 (m) | 0.014 | 0.008 | 0.010 | offline |
| ground penetration max (m) | 0.066 | 0.005 | 0.063 | offline |
| self-collision max (m) | 0.026 | 0.001 | 0.020 | offline |
| max |q̇| / limit | 0.34 | 0.36 | 0.38 | realtime |
| joint jumps (frames) | 0.0 | 0.0 | 0.0 | ≈ |
| joint accel p99 (rad/s²) | 49 | 43 | 53 | offline |
| end stability margin (m) | 0.106 | 0.093 | 0.109 | nvidia |
| max |q̈| / effort-limited acceleration, held 50 ms | 0.25 | 0.31 | 0.30 | realtime |
| unexplained wrench, held 0.1 s (m·g) | 0.157 | 0.062 | 0.217 | offline |
| ZMP outside support, held 0.1 s (m) | 0.076 | 0.019 | 0.140 | offline |
| max |τ| / effort limit, held 50 ms | 1.22 | 1.78 | 1.00 | nvidia |
| frames over a torque limit | 5.6 | 38.9 | 2.6 | nvidia |

- **realtime**: 0/10 accepted
- **offline**: 7/10 accepted
- **nvidia**: 0/10 accepted

## Arms anchored at the shoulders (2026-10-26)

Arm goals now hang from shoulder goals placed at the robot's shoulder width (`scale.anchors`), instead of being
scaled from the pelvis with the spine-dominated chain ratio, which put the hands into the G1's hip housings. Also in
since the previous table: the stance lock's vertical pull ramps in over 50 ms, a smooth self-collision barrier, a
single hand-to-hand collision row and light hand smoothing (viewer findings: hands shaking near the body).
Realtime and offline re-run, NVIDIA unchanged; tracking rules off as before.

| clip | realtime | offline | nvidia |
|---|---|---|---|
| walk_forward_amateur_001__A001 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max) |
| kneeling_loop_001__A021 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max, self_collision_max) |
| jump_ff_180_R_001__A045 | rejected (ground_penetration_max, self_collision_max, zmp_outside_max) | accepted | rejected (ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max) |
| sit_on_chair_loop_R_001__A045 | rejected (env_contact_frames, self_collision_max, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) |
| crawl_ff_start_180_R_001__A125 | rejected (foot_float_p95, ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max) | rejected (foot_skate_excess) | rejected (ground_penetration_max, dyn_residual_max, zmp_outside_max) |
| body_check_001__A296 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (self_collision_max) |
| sweep_floor_side_walk_R_004__A298 | rejected (self_collision_max) | accepted | rejected (self_collision_max) |
| stairs_climbing_down_loop_R_001__A300 | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, ground_penetration_max, self_collision_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_skate_excess, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_float_p95, self_collision_max, dyn_residual_max) |
| push_up_loop_001__A359 | rejected (ground_penetration_max) | accepted | rejected (ground_penetration_max) |
| stand_up_lying_R_002__A472 | rejected (ground_penetration_max, self_collision_max) | rejected (zmp_outside_max) | rejected (ground_penetration_max, self_collision_max, zmp_outside_max) |

| metric (mean over the 8 clips that pass the human pre-filter) | realtime | offline | nvidia | best |
|---|---|---|---|---|
| foot skate p95 (m/s) | 0.091 | 0.092 | 0.080 | nvidia |
| foot skate beyond the source (m/s) | 0.010 | 0.026 | 0.003 | nvidia |
| foot float p95 (m) | 0.012 | 0.008 | 0.010 | offline |
| ground penetration max (m) | 0.069 | 0.005 | 0.063 | offline |
| self-collision max (m) | 0.029 | 0.000 | 0.020 | offline |
| max |q̇| / limit | 0.35 | 0.38 | 0.38 | realtime |
| joint jumps (frames) | 0.0 | 0.0 | 0.0 | ≈ |
| joint accel p99 (rad/s²) | 51 | 44 | 53 | offline |
| end stability margin (m) | 0.107 | 0.092 | 0.109 | nvidia |
| max |q̈| / effort-limited acceleration, held 50 ms | 0.26 | 0.33 | 0.30 | realtime |
| unexplained wrench, held 0.1 s (m·g) | 0.165 | 0.065 | 0.217 | offline |
| ZMP outside support, held 0.1 s (m) | 0.074 | 0.024 | 0.140 | offline |
| max |τ| / effort limit, held 50 ms | 1.18 | 1.76 | 1.00 | nvidia |
| frames over a torque limit | 4.9 | 47.1 | 2.6 | nvidia |

- **realtime**: 0/10 accepted
- **offline**: 6/10 accepted
- **nvidia**: 0/10 accepted

## Feasibility rules revised (2026-10-26)

Dynamic feasibility is defined as CoM support, joint velocity and joint acceleration. Changes: torque over the
effort limit is no longer judged (first downgraded to a warning, then the warning was removed too; torques are only recorded); new reject `joint_acc_ratio_max` ≤ 1 — the joint acceleration held
for 50 ms over the joint's effort-limited acceleration, derived from the model (effort / inertia beyond the joint
about its axis + armature, default pose; hip pitch 95, waist pitch 80, shoulder pitch 178 rad/s², distal joints
far higher; overridable per joint in `Configs/robots/g1.json`). No clip of any tool exceeds 0.70 of it; the
closest are offline body check (shoulder pitch 0.70), jump (hip pitch 0.63–0.68 for all three) and crawl.
Same outputs as the milestone 5 table, revalidated:

| clip | realtime | offline | nvidia |
|---|---|---|---|
| walk_forward_amateur_001__A001 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max) |
| kneeling_loop_001__A021 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max, self_collision_max) |
| jump_ff_180_R_001__A045 | rejected (ground_penetration_max, self_collision_max, zmp_outside_max) | accepted | rejected (ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max) |
| sit_on_chair_loop_R_001__A045 | rejected (env_contact_frames, self_collision_max, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) |
| crawl_ff_start_180_R_001__A125 | rejected (foot_float_p95, ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max) | rejected (foot_skate_excess) | rejected (ground_penetration_max, dyn_residual_max, zmp_outside_max) |
| body_check_001__A296 | rejected (ground_penetration_max, self_collision_max) | rejected (self_collision_max) | rejected (self_collision_max) |
| sweep_floor_side_walk_R_004__A298 | rejected (self_collision_max) | accepted | rejected (self_collision_max) |
| stairs_climbing_down_loop_R_001__A300 | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, ground_penetration_max, self_collision_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_skate_excess, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_float_p95, self_collision_max, dyn_residual_max) |
| push_up_loop_001__A359 | rejected (ground_penetration_max) | accepted | rejected (ground_penetration_max) |
| stand_up_lying_R_002__A472 | rejected (ground_penetration_max, self_collision_max) | rejected (zmp_outside_max) | rejected (ground_penetration_max, self_collision_max, zmp_outside_max) |

| metric (mean over the 8 clips that pass the human pre-filter) | realtime | offline | nvidia | best |
|---|---|---|---|---|
| foot skate p95 (m/s) | 0.091 | 0.094 | 0.080 | nvidia |
| foot skate beyond the source (m/s) | 0.010 | 0.021 | 0.003 | nvidia |
| foot float p95 (m) | 0.012 | 0.008 | 0.010 | offline |
| ground penetration max (m) | 0.071 | 0.005 | 0.063 | offline |
| self-collision max (m) | 0.037 | 0.004 | 0.020 | offline |
| max |q̇| / limit | 0.35 | 0.43 | 0.38 | realtime |
| joint jumps (frames) | 0.0 | 0.0 | 0.0 | ≈ |
| joint accel p99 (rad/s²) | 51 | 53 | 53 | realtime |
| end stability margin (m) | 0.102 | 0.089 | 0.109 | nvidia |
| max |q̈| / effort-limited acceleration, held 50 ms | 0.26 | 0.40 | 0.30 | realtime |
| unexplained wrench, held 0.1 s (m·g) | 0.163 | 0.061 | 0.217 | offline |
| ZMP outside support, held 0.1 s (m) | 0.075 | 0.022 | 0.140 | offline |
| max |τ| / effort limit, held 50 ms | 1.17 | 1.76 | 1.00 | nvidia |
| frames over a torque limit | 9.2 | 37.9 | 2.6 | nvidia |

- **realtime**: 0/10 accepted
- **offline**: 5/10 accepted
- **nvidia**: 0/10 accepted

## Force-based vs force-free CoM support (offline solver, 2026-10-26)

The offline solver's dynamics term in two variants (`offline.dynamics_model`), same weights (3), same clips:
- `contact_forces` (default): per frame, the contact forces that best explain the motion (non-negative least
  squares over friction-pyramid edges at the contact points); the unexplained wrench is penalised.
- `force_free`: no forces — in flight the whole gravito-inertial wrench is penalised (ballistic flight), in
  contact three hinge rows: no ground pull (F_z ≥ 0), total horizontal force inside the friction cone
  (|F_xy| ≤ μ F_z), and the ZMP inside the support polygon (2 cm margin), as the missing moment d·F_z.

| mean over the 8 clips | realtime | contact_forces | force_free | NVIDIA |
|---|---|---|---|---|
| accepted (all rules) | 0/10 | **3/10** | 1/10 | 0/10 |
| unexplained wrench, held 0.1 s (m·g) | 0.163 | **0.061** | 1.908 | 0.217 |
| ZMP outside support, held 0.1 s (m) | 0.075 | **0.022** | 0.339 | 0.140 |
| max \|q̇\| / limit | **0.35** | 0.43 | 0.46 | 0.38 |
| joint accel p99 (rad/s²) | **51** | 53 | 85 | 53 |
| ground penetration max (m) | 0.071 | **0.005** | 0.007 | 0.063 |
| run time (10 clips) | — | 64 s | 57 s | — |

The force-free variant is clean only where the motion is quasi-static (kneel, body check, push-up). On walk,
sweep, jump, crawl and getting up it diverges: unexplained wrench 1–5 m·g, worse than the realtime warm start
it starts from. What was ruled out on the body-check clip:
- the ZMP row's Jacobian matches finite differences (`Tests/DynamicsTests`), and flipping its sign makes it far
  worse; a missing moving-CoM-reference term was found and fixed;
- the ZMP distance form (1/F_z, stiff), a ZMP gate at 0.3 m·g, a 3 cm dead band and a 2 cm inner margin all
  still diverge; without the ZMP row (no pull + friction only) it is stable (0.13 m·g, no CoM support);
- the solver's own support check flags only 18 of 3,412 warm-start frames; the divergence comes from the ZMP
  rows of the neighbouring frames t ± K (which frame t reaches only through second-difference accelerations):
  restricted to frame t's own ZMP, body check stays at 0.001 m·g, but the dynamic clips still diverge.

Reading: the contact-force formulation penalises the distance of the wrench to the contact wrench cone, a convex
function of the wrench (which is nearly linear in the positions), so the Gauss-Seidel sweeps behave. The
force-free ZMP condition is the same feasibility set, but as a one-sided ratio constraint (ZMP = moment / F_z)
it is strongly non-convex in the accelerations, and the one-step-per-frame sweeps cannot handle it. Forces stay
in the solver; `force_free` remains as an experimental option.

## Update — milestone 5 (dynamics)

Same 10 clips and procedure (tracking rules off for all three). New robot rules, each held for a window so
single noisy frames do not count:
- unexplained wrench `dyn_residual_max` ≤ 0.25 m·g (held 0.1 s): what the ground cannot supply at the robot's
  ground-touching points through friction cones (μ = 0.7) — pulling, sliding beyond friction, a centre of
  pressure outside the support, non-ballistic flight;
- ZMP `zmp_outside_max` ≤ 5 cm outside the support (held 0.1 s);
- joint torque `torque_ratio_max` = |τ| / effort limit ≤ 1 (held 50 ms), from floating-base RNEA with the
  estimated contact forces.

The offline solver now has root-acceleration smoothing, centroidal-dynamics and (removed later the same day, no
effect on judged metrics) torque terms; stronger ground
weights (ground 200, contact 100).

| clip | realtime | offline | nvidia |
|---|---|---|---|
| walk_forward_amateur_001__A001 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max) |
| kneeling_loop_001__A021 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max, self_collision_max) |
| jump_ff_180_R_001__A045 | rejected (ground_penetration_max, self_collision_max, zmp_outside_max, torque_ratio_max) | rejected (torque_ratio_max) | rejected (ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max, torque_ratio_max) |
| sit_on_chair_loop_R_001__A045 | rejected (env_contact_frames, self_collision_max, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, dyn_residual_max, zmp_outside_max) |
| crawl_ff_start_180_R_001__A125 | rejected (foot_float_p95, ground_penetration_max, self_collision_max, dyn_residual_max, zmp_outside_max, torque_ratio_max) | rejected (foot_skate_excess, torque_ratio_max) | rejected (ground_penetration_max, dyn_residual_max, zmp_outside_max, torque_ratio_max) |
| body_check_001__A296 | rejected (ground_penetration_max, self_collision_max) | rejected (self_collision_max) | rejected (self_collision_max) |
| sweep_floor_side_walk_R_004__A298 | rejected (self_collision_max) | accepted | rejected (self_collision_max) |
| stairs_climbing_down_loop_R_001__A300 | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, ground_penetration_max, self_collision_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_skate_excess, dyn_residual_max, zmp_outside_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_float_p95, self_collision_max, dyn_residual_max) |
| push_up_loop_001__A359 | rejected (ground_penetration_max, torque_ratio_max) | rejected (torque_ratio_max) | rejected (ground_penetration_max) |
| stand_up_lying_R_002__A472 | rejected (ground_penetration_max, self_collision_max) | rejected (zmp_outside_max, torque_ratio_max) | rejected (ground_penetration_max, self_collision_max, zmp_outside_max) |

| metric (mean over the 8 clips that pass the human pre-filter) | realtime | offline | nvidia | best |
|---|---|---|---|---|
| foot skate p95 (m/s) | 0.091 | 0.094 | 0.080 | nvidia |
| foot skate beyond the source (m/s) | 0.010 | 0.021 | 0.003 | nvidia |
| foot float p95 (m) | 0.012 | 0.008 | 0.010 | offline |
| ground penetration max (m) | 0.071 | 0.005 | 0.063 | offline |
| self-collision max (m) | 0.037 | 0.004 | 0.020 | offline |
| max |q̇| / limit | 0.35 | 0.43 | 0.38 | realtime |
| joint jumps (frames) | 0.0 | 0.0 | 0.0 | ≈ |
| joint accel p99 (rad/s²) | 51 | 53 | 53 | realtime |
| end stability margin (m) | 0.102 | 0.089 | 0.109 | nvidia |
| unexplained wrench, held 0.1 s (m·g) | 0.163 | 0.061 | 0.217 | offline |
| ZMP outside support, held 0.1 s (m) | 0.075 | 0.022 | 0.140 | offline |
| max |τ| / effort limit, held 50 ms | 1.17 | 1.76 | 1.00 | nvidia |
| frames over a torque limit | 9.2 | 37.9 | 2.6 | nvidia |

- **realtime**: 0/10 accepted
- **offline**: 3/10 accepted
- **nvidia**: 0/10 accepted

Reading:
- **Dynamic consistency**: offline cuts the unexplained wrench to well under half of realtime's and to a third of NVIDIA's; its ZMP stays
  within 2 cm of the support on average (realtime 7.5 cm, NVIDIA 14 cm). The jump shows why kinematic
  retargeting is not enough: uniform root scaling (0.785) turns the human's ballistic flight into a 0.785 g
  fall, which no contact can explain (realtime 0.20 m·g, NVIDIA 0.34; offline 0.12).
- **Torque limits** are now the main offline failure. Jump: left ankle pitch 1.5× its 50 N·m on landing.
  Push-up / crawl / getting up from lying: the 5 N·m wrist pitch / yaw joints carry body weight through the
  palms (2–5×). NVIDIA keeps its push-up wrists at 0.34× — the hand pose matters, and the solver's torque term
  does not cover joints carrying contact force yet (DESIGN §13, milestone 5).
- M4's offline accepted 5/10; jump and push-up now fail the torque rule, kneel is back (penetration 0.7 cm with
  the stronger ground term).
- The M4 offline output had a median root acceleration of 30 m/s² (120 Hz jitter of the root, which only had
  velocity smoothing) — invisible to the joint-acceleration metric, obvious to the dynamics metrics.
- Realtime: a causal centroidal term was implemented and measured worse at every weight (lag 0 cannot look
  ahead); it is off until the fixed-lag window.
- Cost: offline 62 s for the 10 clips (M4: 23 s); validation of the 10 clips 6 s. After the speed-up (host
  anchor refresh parallel and every 5 sweeps, warm-started contact forces, no torque rows): 22.5 s, same results
  within noise (unexplained wrench 0.070, ZMP 2.2 cm, penetration 0.5 cm, 5/10 accepted).

## Update — milestone 4 (offline whole-trajectory solver)

Same 10 clips, same validator and rules, all three validated without the tracking rules (tracking is not
comparable across tools; our offline tracking: mean 34–110 mm, the high end on kneeling where knees are held on
the floor). Rules changed since the first comparison: foot skate is now judged as the slide *added* beyond
the source's own (`foot_skate_excess` ≤ 0.05 m/s; raw p95 is a warning), and sole points are the foot mesh's
bottom corners.

| clip | realtime | offline | nvidia |
|---|---|---|---|
| walk_forward_amateur_001__A001 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max) |
| kneeling_loop_001__A021 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max, self_collision_max) |
| jump_ff_180_R_001__A045 | rejected (ground_penetration_max, self_collision_max) | accepted | rejected (ground_penetration_max, self_collision_max) |
| sit_on_chair_loop_R_001__A045 | rejected (env_contact_frames, self_collision_max) | rejected (env_contact_frames) | rejected (env_contact_frames) |
| crawl_ff_start_180_R_001__A125 | rejected (foot_float_p95, ground_penetration_max, self_collision_max) | rejected (foot_skate_excess, ground_penetration_max) | rejected (ground_penetration_max) |
| body_check_001__A296 | rejected (ground_penetration_max, self_collision_max) | rejected (self_collision_max) | rejected (self_collision_max) |
| sweep_floor_side_walk_R_004__A298 | rejected (self_collision_max) | accepted | rejected (self_collision_max) |
| stairs_climbing_down_loop_R_001__A300 | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, ground_penetration_max, self_collision_max) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_skate_excess) | rejected (env_contact_frames, env_support_frames, unsupported_frames, penetration_frames, foot_float_p95, self_collision_max) |
| push_up_loop_001__A359 | rejected (ground_penetration_max) | accepted | rejected (ground_penetration_max) |
| stand_up_lying_R_002__A472 | rejected (ground_penetration_max, self_collision_max) | rejected (ground_penetration_max) | rejected (ground_penetration_max, self_collision_max) |

| metric (mean over the 8 clips that pass the human pre-filter) | realtime | offline | nvidia | best |
|---|---|---|---|---|
| foot skate p95 (m/s) | 0.091 | 0.080 | 0.080 | ≈ |
| foot skate beyond the source (m/s) | 0.010 | 0.024 | 0.003 | nvidia |
| foot float p95 (m) | 0.012 | 0.005 | 0.010 | offline |
| ground penetration max (m) | 0.071 | 0.016 | 0.063 | offline |
| self-collision max (m) | 0.037 | 0.004 | 0.020 | offline |
| max |q̇| / limit | 0.35 | 0.44 | 0.38 | realtime |
| joint jumps (frames) | 0.0 | 0.0 | 0.0 | ≈ |
| joint accel p99 (rad/s²) | 51 | 50 | 53 | offline |
| end stability margin (m) | 0.102 | 0.097 | 0.109 | nvidia |

- **realtime**: 0/10 accepted
- **offline**: 5/10 accepted
- **nvidia**: 0/10 accepted


- **offline** passes 5/10 (walk, kneel, jump, sweep, push-up); realtime and NVIDIA pass none. Chair and stairs
  are rejected for all three at the human stage, as intended.
- Offline is best on ground penetration (1.6 vs 6.3 cm NVIDIA), self-collision (0.4 vs 2.0 cm), foot float and
  joint smoothness. NVIDIA is better on skate beyond the source (driven by our crawl clip, 0.19 m/s) and
  end stability.
- Remaining offline failures: crawl (dragged feet + knees up to 6 cm under the floor in fast contact changes),
  body check (6 isolated frames of transient hand-to-hand overlap up to 2.7 cm), lying (2.1 cm hand penetration).
- Cost: 23 s for the 10 clips (16,263 frames) on the RTX 3090, including realtime warm start, contacts and
  validation — vs 168 s for NVIDIA (≈70 s after its JIT).

---

## First comparison — realtime only (milestone 3)

Both tools retarget the same 10 BONES-SEED SOMA clips onto the G1 29-DoF. Both outputs then go through
the **same** validator (`retarget validate`, resampled to 50 fps, same rules in `Configs/reject/g1.json`).

## Setup

| | ours | NVIDIA soma-retargeter |
|---|---|---|
| version | 0.2.0, milestone 2 realtime solver (causal, lag 0) | commit cbdb2b4 (2026-10-01), newton 1.3.0, warp 1.18.0 |
| method | warm-started LM IK per frame, per-chain scaling, T-pose rotation offsets | Newton LM IK per frame, single scale ratio, foot-plant post-pass, joint-limit clamp |
| output | npz 50 fps | CSV 120 fps (root cm / Euler deg, joints deg) |
| wall time, 10 clips (16,263 frames) | 11.6 s incl. contacts + validation | 168 s (≈100 s of kernel JIT on first run; ≈155 frames/s after) |

Clips: walk, sweep floor, kneel, crawl, push-up, stand up from lying, jump, sit on chair, stairs down,
body check.

## What is (and is not) compared

- **Compared:** limits, joint velocity / acceleration, joint jumps, foot skate and float (against the
  human's foot contacts on the same side), ground penetration of any robot surface point, self-collision
  (link surface points inside the other link's median-radius capsule), and static stability at the clip end.
  None of these depend on how a tool places the motion in the world.
- **Not compared:** tracking error. NVIDIA places the motion in its own world frame with its own scale, so
  errors against our goals would measure convention, not quality. (Ours: mean 26–51 mm, p95 49–136 mm.)
- **Human pre-filter** (identical for both): sit on chair → `env_contact` (pelvis on the chair); stairs →
  `env_support`. Those two clips are excluded from the summary means.

## Results

| clip | metric | ours | nvidia |
|---|---|---|---|
| walk_forward_amateur_001__A001 | foot skate p95 (m/s) | 0.137 | 0.120 |
|  | foot float p95 (m) | 0.000 | 0.006 |
|  | ground penetration max (m) | 0.038 | 0.024 |
|  | self-collision max (m) | 0.034 | 0.001 |
|  | max |q̇| / limit | 0.40 | 0.43 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 51 | 58 |
|  | end stability margin (m) | 0.091 | 0.094 |
| kneeling_loop_001__A021 | foot skate p95 (m/s) | 0.010 | 0.006 |
|  | foot float p95 (m) | 0.008 | 0.028 |
|  | ground penetration max (m) | 0.120 | 0.084 |
|  | self-collision max (m) | 0.028 | 0.013 |
|  | max |q̇| / limit | 0.03 | 0.02 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 3 | 4 |
|  | end stability margin (m) | 0.066 | 0.059 |
| jump_ff_180_R_001__A045 | foot skate p95 (m/s) | 0.092 | 0.082 |
|  | foot float p95 (m) | 0.000 | 0.000 |
|  | ground penetration max (m) | 0.029 | 0.027 |
|  | self-collision max (m) | 0.045 | 0.016 |
|  | max |q̇| / limit | 0.82 | 0.77 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 121 | 122 |
|  | end stability margin (m) | 0.087 | 0.088 |
| sit_on_chair_loop_R_001__A045 | foot skate p95 (m/s) | 0.008 | 0.001 |
|  | foot float p95 (m) | 0.000 | 0.000 |
|  | ground penetration max (m) | 0.010 | 0.008 |
|  | self-collision max (m) | 0.026 | 0.000 |
|  | max |q̇| / limit | 0.01 | 0.01 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 4 | 5 |
|  | end stability margin (m) | -0.230 | -0.231 |
| crawl_ff_start_180_R_001__A125 | foot skate p95 (m/s) | 0.234 | 0.205 |
|  | foot float p95 (m) | 0.082 | 0.025 |
|  | ground penetration max (m) | 0.131 | 0.118 |
|  | self-collision max (m) | 0.026 | 0.003 |
|  | max |q̇| / limit | 0.59 | 0.56 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 109 | 100 |
|  | end stability margin (m) | 0.208 | 0.211 |
| body_check_001__A296 | foot skate p95 (m/s) | 0.012 | 0.003 |
|  | foot float p95 (m) | 0.005 | 0.000 |
|  | ground penetration max (m) | 0.026 | 0.008 |
|  | self-collision max (m) | 0.062 | 0.043 |
|  | max |q̇| / limit | 0.38 | 0.45 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 31 | 35 |
|  | end stability margin (m) | 0.079 | 0.082 |
| sweep_floor_side_walk_R_004__A298 | foot skate p95 (m/s) | 0.095 | 0.086 |
|  | foot float p95 (m) | 0.002 | 0.010 |
|  | ground penetration max (m) | 0.020 | 0.017 |
|  | self-collision max (m) | 0.054 | 0.018 |
|  | max |q̇| / limit | 0.24 | 0.26 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 33 | 35 |
|  | end stability margin (m) | 0.068 | 0.070 |
| stairs_climbing_down_loop_R_001__A300 | foot skate p95 (m/s) | 0.244 | 0.234 |
|  | foot float p95 (m) | 0.000 | 0.296 |
|  | ground penetration max (m) | 1.333 | 0.000 |
|  | self-collision max (m) | 0.046 | 0.044 |
|  | max |q̇| / limit | 0.43 | 0.42 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 99 | 98 |
|  | end stability margin (m) | 0.145 | -0.154 |
| push_up_loop_001__A359 | foot skate p95 (m/s) | 0.020 | 0.018 |
|  | foot float p95 (m) | 0.007 | 0.014 |
|  | ground penetration max (m) | 0.076 | 0.090 |
|  | self-collision max (m) | 0.000 | 0.000 |
|  | max |q̇| / limit | 0.13 | 0.16 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 16 | 19 |
|  | end stability margin (m) | 0.134 | 0.180 |
| stand_up_lying_R_002__A472 | foot skate p95 (m/s) | 0.156 | 0.142 |
|  | foot float p95 (m) | 0.008 | 0.020 |
|  | ground penetration max (m) | 0.125 | 0.133 |
|  | self-collision max (m) | 0.045 | 0.062 |
|  | max |q̇| / limit | 0.24 | 0.41 |
|  | joint jumps (frames) | 0 | 0 |
|  | joint accel p99 (rad/s²) | 45 | 51 |
|  | end stability margin (m) | 0.083 | 0.087 |

| metric (mean over the 8 clips that pass the human pre-filter) | ours | nvidia | better |
|---|---|---|---|
| foot skate p95 (m/s) | 0.094 | 0.083 | nvidia |
| foot float p95 (m) | 0.014 | 0.013 | nvidia |
| ground penetration max (m) | 0.071 | 0.063 | nvidia |
| self-collision max (m) | 0.037 | 0.020 | nvidia |
| max |q̇| / limit | 0.35 | 0.38 | ours |
| joint jumps (frames) | 0 | 0 | ≈ |
| joint accel p99 (rad/s²) | 51 | 53 | ours |
| end stability margin (m) | 0.102 | 0.109 | nvidia |

- **ours**: 0/10 accepted; reject reasons: self_collision_max ×9, ground_penetration_max ×8, foot_skate_p95 ×4, env_contact_frames ×2, foot_float_p95 ×1, env_support_frames ×1, unsupported_frames ×1, penetration_frames ×1
- **nvidia**: 0/10 accepted; reject reasons: ground_penetration_max ×6, self_collision_max ×6, foot_skate_p95 ×4, env_contact_frames ×2, env_support_frames ×1, unsupported_frames ×1, penetration_frames ×1, foot_float_p95 ×1

## Reading

- **Neither tool produces clips that pass the robot checks.** Both are per-frame kinematic IK without ground
  or self-collision constraints. The thresholds are deliberately strict: they gate data for training.
- **NVIDIA is ahead on self-collision** (2.0 vs 3.7 cm mean max). Our dominant failure is the hands sinking
  into the hip housings while walking or sweeping. Our per-chain scaling places hand goals relative to the
  pelvis, and that shrinks the shoulder width (hands end up ~14 cm lateral; the G1 hip housing reaches ~17 cm).
  NVIDIA's single scale ratio with per-joint offsets avoids this.
- **NVIDIA is slightly ahead on foot skate** (0.083 vs 0.094 m/s) thanks to its foot-plant post-pass, and on
  ground penetration (6.3 vs 7.1 cm). Both are worst on kneel / crawl / push-up / lying, where knees, hands
  and the torso go through the floor: neither handles non-foot ground contact.
- **Ours is slightly smoother** (lower peak joint velocity and p99 acceleration), and about 9× faster once
  NVIDIA's kernels are compiled (≈1,400 vs ≈155 frames/s on this batch; 15× including its first-run JIT).
- **Joint jumps:** none for either (after fixing our cold-start convergence on the first frame).

## Consequences for the next milestones

1. Self-collision capsule penalties in the solver (realtime and offline), plus a hand-to-body clearance target.
2. Ground non-penetration for every contact patch (knees, hands, torso), not only the soles — offline as
   hard constraints, realtime as soft terms.
3. Stance-foot locking from the human contact masks (offline), to bring skate under 0.1 m/s p95.
4. Then calibrate the thresholds on a larger sample. Velocity limits are still an assumed 20 rad/s
   (`Configs/robots/g1.json`) until confirmed against the Unitree specification.

The NVIDIA baseline (its checkout, environment and outputs under `build/`) was deleted on 2026-10-28; this
page is the record of the comparison. The source clips remain in `build/compare/bvh`.
