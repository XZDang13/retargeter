from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping


PHUMA_SMPLX_FOOT_VERTEX_INDICES: dict[str, list[int]] = {
    "left_toe_indices": [
        5773,
        5781,
        5782,
        5791,
        5793,
        5805,
        5808,
        5816,
        5817,
        5830,
        5831,
        5859,
        5860,
        5906,
        5907,
        5908,
        5909,
        5912,
        5914,
        5915,
        5916,
        5917,
    ],
    "left_heel_indices": [
        8888,
        8889,
        8891,
        8909,
        8910,
        8911,
        8913,
        8914,
        8915,
        8916,
        8917,
        8918,
        8919,
        8920,
        8921,
        8922,
        8923,
        8924,
        8925,
        8929,
        8930,
        8934,
    ],
    "right_toe_indices": [
        8467,
        8475,
        8476,
        8485,
        8487,
        8499,
        8502,
        8510,
        8511,
        8524,
        8525,
        8553,
        8554,
        8600,
        8601,
        8602,
        8603,
        8606,
        8608,
        8609,
        8610,
        8611,
    ],
    "right_heel_indices": [
        8676,
        8677,
        8679,
        8697,
        8698,
        8699,
        8701,
        8702,
        8703,
        8704,
        8705,
        8706,
        8707,
        8708,
        8709,
        8710,
        8711,
        8712,
        8713,
        8714,
        8715,
        8716,
    ],
}


@dataclass
class LowPassConfig:
    enabled: bool = True
    mode: str = "offline_zero_phase"
    position_cutoff_hz: float = 6.0
    rotation_cutoff_hz: float = 6.0
    root_position_cutoff_hz: float = 3.0
    root_rotation_cutoff_hz: float = 6.0
    order: int = 4

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "LowPassConfig":
        return _dataclass_from_dict(cls, data)


@dataclass
class GroundConfig:
    enabled: bool = True
    method: str = "majority_vote"
    height_bin_size: float = 0.01
    candidate_lower_percent: float = 30.0
    fixed_ground_height: float = 0.0
    foot_vertex_indices: dict[str, list[int]] = field(
        default_factory=lambda: copy.deepcopy(PHUMA_SMPLX_FOOT_VERTEX_INDICES)
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "GroundConfig":
        return _dataclass_from_dict(cls, data)


@dataclass
class ContactConfig:
    enabled: bool = True
    height_threshold: float = 0.06
    velocity_threshold: float = 0.25
    score_height_sigma: float = 0.04
    score_velocity_sigma: float = 0.20
    binary_threshold: float = 0.5
    smooth_contact: bool = True
    smooth_window: int = 3
    foot_vertex_indices: dict[str, list[int]] = field(
        default_factory=lambda: copy.deepcopy(PHUMA_SMPLX_FOOT_VERTEX_INDICES)
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ContactConfig":
        return _dataclass_from_dict(cls, data)


@dataclass
class PreprocessConfig:
    lowpass: LowPassConfig = field(default_factory=LowPassConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PreprocessConfig":
        data = data or {}
        return cls(
            lowpass=LowPassConfig.from_dict(data.get("lowpass")),
            ground=GroundConfig.from_dict(data.get("ground")),
            contact=ContactConfig.from_dict(data.get("contact")),
        )


DEFAULT_PREPROCESS_CONFIG_PATH = Path(__file__).parent / "configs" / "default_preprocess.yaml"


def load_preprocess_config(path: str | Path | None = None) -> PreprocessConfig:
    config_path = Path(path) if path is not None else DEFAULT_PREPROCESS_CONFIG_PATH
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load preprocess YAML configs.") from exc

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return PreprocessConfig.from_dict(data)


def _dataclass_from_dict(cls: type, data: Mapping[str, Any] | None):
    if data is None:
        return cls()
    valid_fields = {f.name for f in fields(cls)}
    kwargs = {k: copy.deepcopy(v) for k, v in data.items() if k in valid_fields}
    return cls(**kwargs)

