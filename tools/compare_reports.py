#!/usr/bin/env python3
"""Compare validation reports of several retargeters (JSON lines from `retarget validate`) as Markdown.

  python3 tools/compare_reports.py build/compare/realtime.jsonl build/compare/offline.jsonl
"""

import json
import sys

METRICS = [
    ("foot_skate_p95", "foot skate p95 (m/s)", 3, False),
    ("foot_skate_excess", "foot skate beyond the source (m/s)", 3, False),
    ("foot_float_p95", "foot float p95 (m)", 3, False),
    ("ground_penetration_max", "ground penetration max (m)", 3, False),
    ("self_collision_max", "self-collision max (m)", 3, False),
    ("velocity_ratio_max", "max |q̇| / limit", 2, False),
    ("joint_jump_frames", "joint jumps (frames)", 1, False),
    ("joint_acc_p99", "joint accel p99 (rad/s²)", 0, False),
    ("end_stability_margin", "end stability margin (m)", 3, True),
    ("joint_acc_ratio_max", "max |q̈| / effort-limited acceleration, held 50 ms", 2, False),
    ("dyn_residual_max", "unexplained wrench, held 0.1 s (m·g)", 3, False),
    ("zmp_outside_max", "ZMP outside support, held 0.1 s (m)", 3, False),
    ("torque_ratio_max", "max |τ| / effort limit, held 50 ms", 2, False),
    ("torque_over_limit_frames", "frames over a torque limit", 1, False),
]


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r["clip"]] = r
    return rows


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = [load(p) for p in sys.argv[1:]]
    labels = [next(iter(r.values()))["label"] or f"run{i}" for i, r in enumerate(runs)]
    clips = sorted(set.intersection(*[set(r) for r in runs]))
    head = " | ".join(labels)
    print(f"| clip | {head} |")
    print("|---|" + "---|" * len(runs))
    for clip in clips:
        cells = []
        for r in runs:
            rep = r[clip]["report"]
            cells.append("accepted" if rep["status"] == "accepted" else "rejected (" + ", ".join(
                x.split(": ", 1)[1].split(" ")[0] for x in rep["reasons"]) + ")")
        print(f"| {clip.split('/')[-1]} | " + " | ".join(cells) + " |")
    valid = [c for c in clips if all(r[c]["report"]["stage"] != "human" for r in runs)]
    print()
    print(f"| metric (mean over the {len(valid)} clips that pass the human pre-filter) | {head} | best |")
    print("|---|" + "---|" * (len(runs) + 1))
    for key, label, digits, higher in METRICS:
        values = []
        for r in runs:
            xs = [r[c]["report"]["metrics"].get(key) for c in valid]
            xs = [x for x in xs if x is not None]
            values.append(sum(xs) / len(xs) if xs else None)
        if any(v is None for v in values):
            continue
        best = (max if higher else min)(range(len(values)), key=lambda i: values[i])
        tie = sum(abs(v - values[best]) < 10 ** -max(digits, 1) for v in values) > 1
        print(f"| {label} | " + " | ".join(f"{v:.{digits}f}" for v in values) + f" | {'≈' if tie else labels[best]} |")
    print()
    for label, r in zip(labels, runs):
        accepted = sum(r[c]["report"]["status"] == "accepted" for c in clips)
        print(f"- **{label}**: {accepted}/{len(clips)} accepted")


if __name__ == "__main__":
    main()
