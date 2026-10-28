# Open-source humanoid retargeting — survey and comparison (2026-10-23)

Purpose: learn from existing tools before building ours (see `../DESIGN.md`). Based on READMEs, configs,
source files and papers; items that could not be confirmed from a primary source are marked *(unverified)*.

## 1. Comparison table

| Tool | License | Solver | Per-frame / trajectory | Scaling | Ground / contact | Physics | Auto-reject | GPU / batch | Speed | G1 23 / 29 |
|---|---|---|---|---|---|---|---|---|---|---|
| [GMR](https://github.com/YanjieZe/GMR) | MIT | mink diff-IK QP, 2 stages | per-frame | per-body × height, uniform root | min-height shift; no contacts | vel limit only (off by default) | no | CPU, single | 35–70 FPS | 29 |
| [NVIDIA soma-retargeter](https://github.com/NVIDIA/soma-retargeter) | Apache-2.0 | LM in CUDA graph, 24 it | per-frame + post-pass | single scalar | human-side foot plants (joints), assumes z = 0 | none | no | GPU, clip batches | not published | 29 |
| [mink](https://github.com/kevinzakka/mink) | Apache-2.0 | diff-IK QP | per-frame | — | collision-pair limits | none | — | CPU | — | — |
| [PyRoki](https://github.com/chungmin99/pyroki) | MIT | jaxls LM / aug. Lagrangian | whole trajectory | scale as variable | contact-gated foot height + skate, heightmap, self-collision | none | no | JAX GPU | — | 29 example |
| PHC / H2O / [ASAP](https://github.com/LeCAR-Lab/ASAP) | CC BY-NC / MIT | Adam on keypoints | whole clip | SMPL β fit | first-frame min-z | sim-to-data via RL policy | policy fail | CPU pool | slow | 23 (from 29) |
| [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) | *(unverified)* | IPOPT | per-frame, arms only | arm scale | none | RNEA gravity FF | — | CPU | 30 Hz | 23 + 29 |
| TWIST / TWIST2 | MIT / *(unverified)* | GMR variant | per-frame | GMR | — | — | — | CPU | < 100 ms e2e | 29 |
| [holosoma / OmniRetarget](https://github.com/amazon-far/holosoma) | Apache-2.0 | SQP (Clarabel), trust region | per-frame | input-side | **hard** SDF non-penetration, foot sticking, interaction mesh | none | no | CPU parallel | — | 29 |
| PHUMA / PhySINK ([paper](https://arxiv.org/html/2510.26236)) | Apache-2.0 *(code URL reported once)* | gradient (Adam) | whole trajectory | SMPL β | **majority-vote ground**, heel/toe vertex contacts | soft limits 98%, skate | **yes, rule-based** | GPU | 0.92 FPS | 29 |
| [SPIDER](https://github.com/facebookresearch/spider) | CC BY-NC | sampling in MuJoCo Warp | trajectory | — | simulated | full sim in loop | object success only | GPU | ~20 FPS | yes |
| [PBHC / KungfuBot](https://github.com/TeleHuman/PBHC) | CC BY-NC | mink IK | per-frame | — | ankle vel + height | — | **CoM–CoP stability** | — | — | 23 |
| KDMR ([paper](https://arxiv.org/html/2603.09956)) | anon. repo | CasADi / Pinocchio | trajectory | — | needs measured GRF | **hard full dynamics, friction, ZMP, torque** | no | CPU | 88–282 s / 800 frames | yes |
| IHMC stability-aware ([paper](https://arxiv.org/html/2510.04353)) | — | LP stability region | per-frame | — | hand contacts | CoM in centroidal region incl. torque | — | CPU | 79–698 µs | — |

## 2. What each family teaches us

**Per-frame IK (GMR, soma-retargeter, mink, xr_teleoperate, holosoma).**
- Warm-started LM / diff-IK at 10–24 iterations is enough for good per-frame quality; soma-retargeter
  shows the CUDA-graph LM pattern we planned for realtime.
- GMR's paper is the key evidence on artifacts: ground penetration (up to 60 cm in PHC data), leg
  self-intersection (26 % vs 100 % sim2sim success) and waist jumps break downstream policies.
- Scaling matters most: per-segment scale with a **uniformly scaled root** avoids induced foot sliding;
  a single global scale (soma, ProtoMotions) is a main source of sliding and self-intersection.
- TWIST2 for teleop: lower body tracked on position + rotation, upper body rotation-only, all
  pelvis-relative — robust to global drift of VR / mocap streams. Reached < 100 ms end-to-end.

**Whole-trajectory optimisation (PyRoki, PhySINK, PHC, VideoMimic).**
- One least-squares problem over the clip with contact-gated foot height and skating, self- and
  ground-collision, limits at ~98 % of range, smoothness. Matches our offline design.
- PyRoki optimises segment scale as a variable — worth having as an option.
- Soft penalties leave residual penetration / sliding; hard constraints (holosoma) still leave ~1 cm
  after linearisation. → Use hard constraints (augmented Lagrangian) plus a final validation.

**Physics / dynamics (KDMR, IHMC, SPIDER, sim-to-data).**
- KDMR is closest to our offline dynamics stage (friction pyramid, ZMP, torque limits) and shows real
  gains over GMR, but relies on measured ground reaction forces — we must infer contact forces instead.
- IHMC shows a centroidal stability check fast enough for teleop.
- Sim-in-the-loop (SPIDER) and policy-based filtering (H2O, ASAP) are strong but expensive and opaque;
  out of scope for v1 (matches the decision to skip simulation validation).

**Data curation (PHUMA, PBHC, GMT, UniTracker, EGM).**
- Two-stage rejection works: human-side pre-filters, then robot-side post-filters.
- Hand-screened small sets can beat large auto-filtered sets (EGM) → thresholds must be calibrated
  and every metric must be stored, not only pass/fail.

**Ground and contacts.**
- Nobody does vertex-based ground estimation for SOMA. soma-retargeter assumes z = 0 and uses joints;
  GMR / PHC use a min-height shift (causes floating / penetration elsewhere in the clip).
- PHUMA: global majority vote over foot vertices (±2.5 cm), one shift per clip.
- HuMoR: DBSCAN over static toe heights; flags terrain when another height cluster > 4 cm persists > 0.25 s.
- **SOMA-X** ([repo](https://github.com/NVlabs/SOMA-X), Apache-2.0) provides a skinned body mesh
  (18k / 4.5k / 612-vertex LODs, sparse LBS weights) for exactly our 78-node skeleton. BONES-SEED
  ships the shape parameters of the uniform skeleton (`soma_shapes/soma_base_fit_mhr_params.npz`).
  → We can pose real body vertices from the BVH. *(Exact alignment with the BVH rest pose unverified —
  first thing to test.)*

## 3. Where our design sits

No existing tool combines: batched GPU realtime + batched whole-trajectory offline + inferred contact-
force dynamics + vertex-based ground / any-body contacts + automatic rejection + G1 23 **and** 29.
Closest pieces: soma-retargeter (GPU per-frame, SOMA input), PyRoki (trajectory LS), KDMR (dynamics),
PHUMA (ground + curation).

## 4. Adopted into DESIGN.md

1. Per-segment scaling with uniform root scale; optional scale-as-variable in offline.
2. Weight-scheduled / two-stage solve (orientations + end-effectors, then all positions).
3. Teleop: pelvis-relative targets; upper body rotation-only option (TWIST2).
4. Smooth inner-limit penalty, limits at 98 % for the solve, hard clamp on output.
5. Offline: zero-phase Butterworth pre-filter (3 Hz root / 6 Hz pose), solve at source 120 Hz.
6. Offline hard constraints (non-penetration, stance lock, self-collision, limits) via augmented Lagrangian;
   LM trust region.
7. Vertex-based ground estimation (majority vote + terrain detection) and any-body contact detection
   with hysteresis.
8. Two-stage auto-reject (human pre-filter + robot post-filter) with published thresholds as defaults;
   boundary checks (first frame reachable, last frame stable); joint-jump detector.
9. Weight tuning on a fixed clip corpus with p95 / worst-clip guards (soma's optimiser idea).
10. Batch robustness: per-clip failure isolation, resume, fps stored in every output.

Not adopted in v1: interaction-mesh Laplacian (useful for object / scene interaction — v2),
sim-in-the-loop validation, learned contact classifiers.

Licences: SPIDER and PBHC are CC BY-NC — ideas only, no code. Others listed are permissive.

## 5. Published reject thresholds (starting defaults, to be calibrated on SOMA)

| Criterion | Threshold | Source |
|---|---|---|
| Root jerk | > 50 m/s³ | PHUMA |
| Foot contact score | < 0.6 | PHUMA |
| Pelvis height (human) | < 0.6 m or > 1.5 m | PHUMA |
| Pelvis / spine1 outside foot support | > 6 cm / > 11 cm | PHUMA |
| CoM–CoP horizontal distance | ≥ 0.1 for ≥ 100 frames, or at first/last frame | PBHC |
| Floating / penetration in contact | > 1 cm | PHUMA metrics |
| Contact foot speed | > 10 cm/s (slide if > 1 cm/s stance, OmniRetarget) | PHUMA / OmniRetarget |
| Joint pos / vel | beyond 98 % of limits | PhySINK |
| Self-collision clearance | < 2 cm (holosoma tolerance) | holosoma |
| Stability margin | < 4 cm | IHMC |
| Tracking failure | mean body error > 0.5 m (sim) | PHC / H2O |
| Clip length | < 10 frames | UniTracker |

## 6. What "physically feasible" means in the literature (2026-10-26)

Five levels appear, usually not all in one work:

| Level | Criterion | Who | Thresholds |
|---|---|---|---|
| 1. Kinematic validity | joint position / velocity within limits; no self-collision or penetration; no joint jumps | PHUMA (PhySINK), OmniRetarget (hard constraints), GMR (artifacts it avoids) | PHUMA: within 98 % of position and velocity limits; OmniRetarget: signed distance ≥ 0, velocity limits as hard constraints |
| 2. Contact consistency | feet on the ground while in contact, not sliding | PHUMA metrics, OmniRetarget metrics | PHUMA: float ≤ 1 cm, penetration ≤ 1 cm, skate < 10 cm/s during contact (scored as % of frames); OmniRetarget: stance = source foot slower than 1 cm/s, foot held fixed |
| 3. Source balance (dataset curation, on the human) | CoM / pelvis over the base of support; smooth root | PHUMA curation; KungfuBot (PBHC) | PHUMA: root jerk < 50 m/s³, pelvis within 6 cm and spine1 within 11 cm of the support hull, contact score > 0.6; KungfuBot: CoM–CoP ground distance < ε per frame, first and last frame stable, longest unstable run < ε_N (values in its appendix) |
| 4. Dynamic feasibility of the robot motion | the base wrench the motion needs lies in the contact wrench cone (friction + centre of pressure inside each contact); torques within limits | Zheng & Yamane (Disney), KDMR, SPARK | Zheng & Yamane: feasibility index = signed distance of the required base wrench to the contact wrench cone (> 0 infeasible); KDMR / SPARK: full rigid-body dynamics, friction pyramids, CoP box per foot, \|τ\| ≤ τ_max as hard constraints (no published pass/fail metric) |
| 5. Trackability (the operational definition) | a tracking policy can follow the motion in simulation | H2O / OmniH2O (sim-to-data: ~8.5k of 10k AMASS clips kept for H1), PHC, GMR, OmniRetarget, BeyondMimic | episode terminates when the anchor (root / torso) height deviates > 0.25 m or orientation drifts (BeyondMimic-style); success = reaching the end |

Observations:
- Kinematic retargeters (GMR, OmniRetarget, PHUMA) explicitly leave dynamics to the RL tracker; their own checks are
  levels 1–2, and dataset curation adds level 3 on the source. OmniRetarget calls its output "kinematically
  feasible"; GMR finds penetration, self-intersection, joint jumps and the start pose predict tracking failure.
- Dynamics-based retargeters (KDMR, SPARK) impose level 4 as constraints and report tracking-policy results, not a
  feasibility score. Zheng & Yamane's index is the closest published metric to a level-4 validator rule.
- Joint acceleration limits are not used as a feasibility criterion by any of these works; acceleration appears as a
  smoothness cost (KDMR) or as root jerk in curation (PHUMA). Torque limits appear only in the optimisation-based
  works (KDMR, SPARK).
- Level 5 is what decides in practice (most works report success rates); it needs a simulator and a tracking policy.

Sources: PHUMA https://arxiv.org/html/2510.26236 · OmniRetarget https://arxiv.org/html/2509.26633v1 ·
GMR https://arxiv.org/html/2510.02252 · KungfuBot https://arxiv.org/html/2506.12851v3 ·
H2O https://arxiv.org/pdf/2403.04436 · ExBody2 https://arxiv.org/html/2412.13196v1 ·
KDMR https://arxiv.org/html/2603.09956 · SPARK https://arxiv.org/html/2603.11480 ·
Zheng & Yamane, "Adapting Human Motions to Humanoid Robots Through Time Warping Based on a General Motion Feasibility
Index" https://la.disneyresearch.com/wp-content/uploads/Adapting-Human-Motions-to-Humanoid-Robots-Through-Time-Warping-Based-on-a-General-Motion-Feasibility-Index-Paper.pdf ·
BeyondMimic https://arxiv.org/pdf/2508.08241 (termination values as reported by TEXEDO https://arxiv.org/pdf/2606.22998
and unitree_rl_lab).

## Sources

GMR https://github.com/YanjieZe/GMR · paper https://arxiv.org/html/2510.02252 ·
soma-retargeter https://github.com/NVIDIA/soma-retargeter (documentation/*.md) ·
mink https://github.com/kevinzakka/mink · PyRoki https://github.com/chungmin99/pyroki ·
PHC https://github.com/ZhengyiLuo/PHC · human2humanoid https://github.com/LeCAR-Lab/human2humanoid ·
ASAP https://github.com/LeCAR-Lab/ASAP · xr_teleoperate https://github.com/unitreerobotics/xr_teleoperate ·
TWIST https://github.com/YanjieZe/TWIST · TWIST2 https://arxiv.org/html/2511.02832v1 ·
holosoma https://github.com/amazon-far/holosoma · OmniRetarget https://arxiv.org/html/2509.26633v1 ·
PHUMA https://arxiv.org/html/2510.26236 · SPIDER https://github.com/facebookresearch/spider ·
PBHC https://github.com/TeleHuman/PBHC · KDMR https://arxiv.org/html/2603.09956 ·
IHMC https://arxiv.org/html/2510.04353 · VideoMimic https://github.com/hongsukchoi/VideoMimic ·
HuMoR https://github.com/davrempe/humor · SOMA-X https://github.com/NVlabs/SOMA-X ·
BONES-SEED https://huggingface.co/datasets/bones-studio/seed ·
SMPL-X vertex ids https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py ·
SMPL-X segmentation https://github.com/Meshcapade/wiki/tree/main/assets/SMPL_body_segmentation/smplx ·
ReActor https://arxiv.org/abs/2605.06593
