"""Stage 1 Newton replay visualization and diagnostics."""

from .config import DEFAULT_VIS_CONFIG_PATH, load_vis_config
from .diagnostic_plots import (
    plot_contact_scores,
    plot_foot_height_and_speed,
    plot_frame_success,
    plot_ik_errors,
    plot_joint_limit_margin,
    plot_joint_positions,
    plot_joint_velocities,
    plot_root_height,
)
from .loaders import export_canonical_human_motion_npz, load_canonical_human_motion_npz, load_preprocess_result_npz
from .newton_replay import (
    DEFAULT_HUMAN_MESH_COLOR,
    DEFAULT_HUMAN_MESH_OFFSET,
    NEWTON_VIEWER_KINDS,
    NewtonReplayResult,
    record_stage1_newton_replay,
    replay_stage1_motion_with_newton,
    stage1_frame_to_ik_state,
    validate_stage1_motion_for_robot,
)

__all__ = [
    "DEFAULT_VIS_CONFIG_PATH",
    "DEFAULT_HUMAN_MESH_COLOR",
    "DEFAULT_HUMAN_MESH_OFFSET",
    "NEWTON_VIEWER_KINDS",
    "NewtonReplayResult",
    "export_canonical_human_motion_npz",
    "load_canonical_human_motion_npz",
    "load_preprocess_result_npz",
    "load_vis_config",
    "plot_contact_scores",
    "plot_foot_height_and_speed",
    "plot_frame_success",
    "plot_ik_errors",
    "plot_joint_limit_margin",
    "plot_joint_positions",
    "plot_joint_velocities",
    "plot_root_height",
    "record_stage1_newton_replay",
    "replay_stage1_motion_with_newton",
    "stage1_frame_to_ik_state",
    "validate_stage1_motion_for_robot",
]
