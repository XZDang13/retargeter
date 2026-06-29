from __future__ import annotations

from pathlib import Path

import numpy as np

from retargeter.newton import load_retargeted_motion_npz
from retargeter.preprocess import CanonicalHumanMotion, FootContactResult, PreprocessResult
from retargeter.refinement import load_refined_motion_npz


PRIMARY_REPLAY_MOTION_NAMES = ("final_motion.npz", "online_motion.npz", "retargeted_motion.npz")
REPLAY_MOTION_PRIORITY = PRIMARY_REPLAY_MOTION_NAMES


def export_canonical_human_motion_npz(
    motion: CanonicalHumanMotion,
    path: Path | str,
    *,
    preprocess_result: PreprocessResult | None = None,
    require_mesh: bool = False,
) -> Path:
    motion.validate()
    if require_mesh and (motion.vertices_w is None or motion.mesh_faces is None):
        raise ValueError("Human replay export requires vertices_w and mesh_faces.")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": np.asarray(float(motion.fps)),
        "body_names": np.asarray(motion.body_names),
        "body_pos_w": np.asarray(motion.body_pos_w, dtype=np.float64),
        "body_quat_xyzw": np.asarray(motion.body_quat_xyzw, dtype=np.float64),
    }
    if motion.vertices_w is not None:
        payload["vertices_w"] = np.asarray(motion.vertices_w, dtype=np.float64)
    if motion.mesh_faces is not None:
        payload["mesh_faces"] = np.asarray(motion.mesh_faces, dtype=np.int32)

    contact = None if preprocess_result is None else preprocess_result.contact
    if contact is not None:
        payload["ground_height"] = np.asarray(float(contact.ground_height))
        for region, score in contact.contact_score.items():
            payload[f"contact_score_{region}"] = np.asarray(score, dtype=np.float64)
        for region, binary in contact.contact_binary.items():
            payload[f"contact_binary_{region}"] = np.asarray(binary, dtype=bool)
        for region, height in contact.foot_height.items():
            payload[f"foot_height_{region}"] = np.asarray(height, dtype=np.float64)
        for region, speed in contact.foot_speed.items():
            payload[f"foot_speed_{region}"] = np.asarray(speed, dtype=np.float64)

    np.savez_compressed(output, **payload)
    return output


def load_canonical_human_motion_npz(path: Path | str) -> CanonicalHumanMotion:
    data = np.load(Path(path), allow_pickle=False)
    required = ["fps", "body_names", "body_pos_w", "body_quat_xyzw"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Canonical human motion npz {path} is missing fields: {missing}.")
    vertices = np.asarray(data["vertices_w"], dtype=np.float64) if "vertices_w" in data else None
    mesh_faces = np.asarray(data["mesh_faces"], dtype=np.int32) if "mesh_faces" in data else None
    motion = CanonicalHumanMotion(
        fps=float(data["fps"]),
        body_names=[str(name) for name in data["body_names"].tolist()],
        body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float64),
        body_quat_xyzw=np.asarray(data["body_quat_xyzw"], dtype=np.float64),
        vertices_w=vertices,
        mesh_faces=mesh_faces,
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


def resolve_replay_motion_path(path: Path | str) -> Path:
    input_path = Path(path)
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Replay input does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Replay input must be a motion npz or output directory, got {input_path}.")
    for name in REPLAY_MOTION_PRIORITY:
        candidate = input_path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find replay motion in {input_path}. Expected one of {list(REPLAY_MOTION_PRIORITY)}."
    )


def default_human_path_for_replay_input(path: Path | str) -> Path | None:
    input_path = Path(path)
    directory = input_path if input_path.is_dir() else input_path.parent
    candidate = directory / "human.npz"
    return candidate if candidate.exists() else None


def load_replay_motion_npz(path: Path | str):
    motion_path = resolve_replay_motion_path(path)
    data = np.load(motion_path, allow_pickle=False)
    if "success" in data.files:
        return load_retargeted_motion_npz(motion_path)
    if "root_delta" in data.files or "joint_delta" in data.files:
        return load_refined_motion_npz(motion_path)
    raise ValueError(
        f"Replay motion {motion_path} is neither retargeted nor refined motion format. "
        "Expected 'success' or refinement delta fields."
    )


def _regions_from_keys(keys: list[str], prefix: str) -> list[str]:
    return sorted(key[len(prefix) :] for key in keys if key.startswith(prefix))
