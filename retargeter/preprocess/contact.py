from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canonical import CanonicalHumanMotion
from .config import ContactConfig


CONTACT_REGIONS = ["left_foot", "right_foot", "left_toe", "right_toe", "left_heel", "right_heel"]
REGION_VERTEX_KEYS = {
    "left_foot": ["left_toe_indices", "left_heel_indices"],
    "right_foot": ["right_toe_indices", "right_heel_indices"],
    "left_toe": ["left_toe_indices"],
    "right_toe": ["right_toe_indices"],
    "left_heel": ["left_heel_indices"],
    "right_heel": ["right_heel_indices"],
}


@dataclass
class FootContactResult:
    contact_score: dict[str, np.ndarray]
    contact_binary: dict[str, np.ndarray]
    foot_height: dict[str, np.ndarray]
    foot_speed: dict[str, np.ndarray]
    ground_height: float
    metadata: dict = field(default_factory=dict)


class FootContactEstimator:
    def __init__(self, config: ContactConfig):
        self.config = config

    def estimate(self, motion: CanonicalHumanMotion, ground_height: float) -> FootContactResult:
        contact_score: dict[str, np.ndarray] = {}
        contact_binary: dict[str, np.ndarray] = {}
        foot_height: dict[str, np.ndarray] = {}
        foot_speed: dict[str, np.ndarray] = {}
        missing_regions: list[str] = []
        sources: dict[str, str] = {}

        for region in CONTACT_REGIONS:
            region_data = self._get_region_motion(motion, region)
            if region_data is None:
                missing_regions.append(region)
                height = np.full((motion.num_frames(),), np.inf, dtype=np.float64)
                speed = np.zeros((motion.num_frames(),), dtype=np.float64)
                score = np.zeros((motion.num_frames(),), dtype=np.float64)
                binary = np.zeros((motion.num_frames(),), dtype=bool)
            else:
                z, xy, source = region_data
                sources[region] = source
                height = z - ground_height
                speed = _horizontal_speed(xy, motion.fps)
                height_for_score = np.maximum(height, 0.0)
                sigma_h = max(float(self.config.score_height_sigma), 1e-9)
                sigma_v = max(float(self.config.score_velocity_sigma), 1e-9)
                height_score = np.exp(-(height_for_score**2) / (2.0 * sigma_h**2))
                speed_score = np.exp(-(speed**2) / (2.0 * sigma_v**2))
                score = np.clip(height_score * speed_score, 0.0, 1.0)
                binary = score >= self.config.binary_threshold
                if self.config.smooth_contact:
                    binary = _smooth_binary(binary, self.config.smooth_window)

            foot_height[region] = height
            foot_speed[region] = speed
            contact_score[region] = score
            contact_binary[region] = binary

        return FootContactResult(
            contact_score=contact_score,
            contact_binary=contact_binary,
            foot_height=foot_height,
            foot_speed=foot_speed,
            ground_height=float(ground_height),
            metadata={
                "regions": list(CONTACT_REGIONS),
                "missing_regions": missing_regions,
                "sources": sources,
                "height_threshold": self.config.height_threshold,
                "velocity_threshold": self.config.velocity_threshold,
            },
        )

    def _get_region_motion(self, motion: CanonicalHumanMotion, region: str) -> tuple[np.ndarray, np.ndarray, str] | None:
        if motion.vertices_w is not None:
            vertex_count = motion.vertices_w.shape[1]
            indices = []
            for key in REGION_VERTEX_KEYS[region]:
                indices.extend(self.config.foot_vertex_indices.get(key, []))
            valid_indices = _valid_indices(indices, vertex_count)
            if valid_indices.size:
                vertices = motion.vertices_w[:, valid_indices, :]
                return np.min(vertices[..., 2], axis=1), np.mean(vertices[..., :2], axis=1), "vertices"

        if region in motion.body_names:
            pos = motion.get_body_pos(region)
            return pos[:, 2], pos[:, :2], "bodies"

        return None


def _horizontal_speed(xy: np.ndarray, fps: float) -> np.ndarray:
    speed = np.zeros((xy.shape[0],), dtype=np.float64)
    if xy.shape[0] <= 1:
        return speed
    diff_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    speed[1:] = diff_speed
    speed[0] = diff_speed[0]
    return speed


def _smooth_binary(binary: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or binary.size <= 1:
        return binary.astype(bool, copy=True)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(binary.astype(np.float64), (radius, radius), mode="edge")
    out = np.zeros_like(binary, dtype=bool)
    for i in range(binary.size):
        out[i] = np.mean(padded[i : i + window]) >= 0.5
    return out


def _valid_indices(indices: list[int], vertex_count: int) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    if arr.size == 0:
        return arr
    return arr[(arr >= 0) & (arr < vertex_count)]

