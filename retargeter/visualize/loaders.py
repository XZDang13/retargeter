from __future__ import annotations

from pathlib import Path

import numpy as np

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult, PreprocessResult


def load_canonical_human_motion_npz(path: Path | str) -> CanonicalHumanMotion:
    data = np.load(Path(path), allow_pickle=False)
    required = ["fps", "body_names", "body_pos_w", "body_quat_xyzw"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Canonical human motion npz {path} is missing fields: {missing}.")
    vertices = np.asarray(data["vertices_w"], dtype=np.float64) if "vertices_w" in data else None
    motion = CanonicalHumanMotion(
        fps=float(data["fps"]),
        body_names=[str(name) for name in data["body_names"].tolist()],
        body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float64),
        body_quat_xyzw=np.asarray(data["body_quat_xyzw"], dtype=np.float64),
        vertices_w=vertices,
        metadata={"loaded_from": str(path)},
    )
    motion.validate()
    return motion


def load_preprocess_result_npz(path: Path | str, motion: CanonicalHumanMotion) -> PreprocessResult | None:
    data = np.load(Path(path), allow_pickle=False)
    regions = _regions_from_keys(data.files, "contact_score_")
    if not regions:
        return None

    contact_score = {}
    contact_binary = {}
    foot_height = {}
    foot_speed = {}
    for region in regions:
        contact_score[region] = np.asarray(data[f"contact_score_{region}"], dtype=np.float64)
        if f"contact_binary_{region}" in data:
            contact_binary[region] = np.asarray(data[f"contact_binary_{region}"], dtype=bool)
        else:
            contact_binary[region] = contact_score[region] >= 0.5
        if f"foot_height_{region}" in data:
            foot_height[region] = np.asarray(data[f"foot_height_{region}"], dtype=np.float64)
        else:
            foot_height[region] = np.zeros_like(contact_score[region])
        if f"foot_speed_{region}" in data:
            foot_speed[region] = np.asarray(data[f"foot_speed_{region}"], dtype=np.float64)
        else:
            foot_speed[region] = np.zeros_like(contact_score[region])

    contact = FootContactResult(
        contact_score=contact_score,
        contact_binary=contact_binary,
        foot_height=foot_height,
        foot_speed=foot_speed,
        ground_height=float(data["ground_height"]) if "ground_height" in data else 0.0,
        metadata={"loaded_from": str(path), "regions": sorted(regions)},
    )
    return PreprocessResult(
        motion=motion,
        ground=None,
        contact=contact,
        warnings=[],
        metadata={"loaded_from": str(path), "contact_available": True},
    )


def _regions_from_keys(keys: list[str], prefix: str) -> list[str]:
    return sorted(key[len(prefix) :] for key in keys if key.startswith(prefix))
