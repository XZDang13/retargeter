from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canonical import CanonicalHumanMotion
from .config import GroundConfig


FOOT_VERTEX_KEYS = [
    "left_toe_indices",
    "left_heel_indices",
    "right_toe_indices",
    "right_heel_indices",
]
FOOT_BODY_NAMES = ["left_foot", "right_foot", "left_toe", "right_toe", "left_heel", "right_heel"]


@dataclass
class GroundEstimate:
    ground_height: float
    confidence: float
    method: str
    metadata: dict = field(default_factory=dict)


class GroundPlaneEstimator:
    def __init__(self, config: GroundConfig):
        self.config = config

    def estimate(self, motion: CanonicalHumanMotion) -> GroundEstimate:
        method = self.config.method
        if method == "fixed":
            return GroundEstimate(
                ground_height=float(self.config.fixed_ground_height),
                confidence=1.0,
                method="fixed",
                metadata={},
            )

        samples, source = self._collect_foot_heights(motion)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            return GroundEstimate(
                ground_height=float(self.config.fixed_ground_height),
                confidence=0.0,
                method="fixed_fallback",
                metadata={"reason": "no_foot_data", "source": source},
            )

        if method == "percentile":
            height = float(np.percentile(samples, self.config.candidate_lower_percent))
            return GroundEstimate(
                ground_height=height,
                confidence=0.5,
                method="percentile",
                metadata={"num_samples": int(samples.size), "source": source},
            )

        if method != "majority_vote":
            raise ValueError(f"Unsupported ground estimation method {method!r}.")

        lower_percent = float(np.clip(self.config.candidate_lower_percent, 0.0, 100.0))
        threshold = np.percentile(samples, lower_percent)
        candidates = samples[samples <= threshold]
        if candidates.size == 0:
            candidates = samples

        bin_size = max(float(self.config.height_bin_size), 1e-6)
        bin_ids = np.round(candidates / bin_size).astype(np.int64)
        unique_bins, counts = np.unique(bin_ids, return_counts=True)
        selected_bin = int(unique_bins[int(np.argmax(counts))])
        ground_height = selected_bin * bin_size
        confidence = float(np.max(counts) / samples.size)

        return GroundEstimate(
            ground_height=float(ground_height),
            confidence=confidence,
            method="majority_vote",
            metadata={
                "num_samples": int(samples.size),
                "num_candidates": int(candidates.size),
                "bin_size": bin_size,
                "source": source,
            },
        )

    def normalize_to_ground(self, motion: CanonicalHumanMotion, ground_height: float) -> CanonicalHumanMotion:
        normalized = motion.copy()
        normalized.body_pos_w[..., 2] -= ground_height
        if normalized.vertices_w is not None:
            normalized.vertices_w[..., 2] -= ground_height
        normalized.metadata["original_ground_height"] = float(ground_height)
        normalized.metadata["normalized_ground_height"] = 0.0
        return normalized

    def _collect_foot_heights(self, motion: CanonicalHumanMotion) -> tuple[np.ndarray, str]:
        if motion.vertices_w is not None:
            vertex_heights = []
            vertex_count = motion.vertices_w.shape[1]
            for key in FOOT_VERTEX_KEYS:
                indices = _valid_indices(self.config.foot_vertex_indices.get(key, []), vertex_count)
                if indices.size:
                    vertex_heights.append(motion.vertices_w[:, indices, 2].reshape(-1))
            if vertex_heights:
                return np.concatenate(vertex_heights), "vertices"

        body_heights = []
        for name in FOOT_BODY_NAMES:
            if name in motion.body_names:
                body_heights.append(motion.get_body_pos(name)[:, 2])
        if body_heights:
            return np.concatenate(body_heights), "bodies"

        return np.empty((0,), dtype=np.float64), "none"


def _valid_indices(indices: list[int], vertex_count: int) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    if arr.size == 0:
        return arr
    return arr[(arr >= 0) & (arr < vertex_count)]
