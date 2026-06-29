from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .canonical import CanonicalHumanMotion
from .config import PreprocessConfig
from .preprocessor import MotionPreprocessor, PreprocessResult
from .resample import resample_smpl_motion
from .smpl_fk import SMPLForwardKinematics
from .smpl_motion import load_smpl_motion


@dataclass
class SMPLPreprocessOutput:
    canonical_motion: CanonicalHumanMotion
    preprocess_result: PreprocessResult
    source_metadata: dict


def run_smpl_preprocess(
    input_path: Path | str,
    preprocess_config: PreprocessConfig,
    *,
    model_type: Literal["smpl", "smplx"] | None = None,
    fps: float | None = None,
    gender: str | None = None,
    smpl_model_dir: Path | str = Path("assets/body_models"),
    device: str = "cpu",
    return_vertices: bool = True,
    target_fps: float | None = None,
) -> SMPLPreprocessOutput:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input motion file does not exist: {input_path}")

    smpl_motion = load_smpl_motion(
        input_path,
        model_type=model_type,
        fps=fps,
        gender=gender,
    )
    if target_fps is not None:
        smpl_motion = resample_smpl_motion(smpl_motion, target_fps)
    model_dir = Path(smpl_model_dir)
    validate_smpl_model_dir(model_dir, smpl_motion.model_type)

    try:
        fk = SMPLForwardKinematics(
            model_dir=model_dir,
            model_type=smpl_motion.model_type,
            gender=gender or smpl_motion.gender,
            device=device,
            foot_vertex_config=preprocess_config.ground.foot_vertex_indices,
        )
        canonical_motion = fk.forward(smpl_motion, return_vertices=return_vertices)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to run SMPL forward kinematics for {input_path}. "
            f"Check --smpl-model-dir, --model-type, --gender, and installed smplx/torch packages."
        ) from exc

    preprocess_result = MotionPreprocessor(preprocess_config).process(canonical_motion)
    source_metadata = {
        "input": str(input_path),
        "mock_mode": False,
        "model_type": smpl_motion.model_type,
        "gender": smpl_motion.gender,
        "smpl_model_dir": str(model_dir),
        "smpl_fk_applied": True,
    }
    if "resample" in smpl_motion.metadata:
        source_metadata["resample"] = dict(smpl_motion.metadata["resample"])
    return SMPLPreprocessOutput(
        canonical_motion=canonical_motion,
        preprocess_result=preprocess_result,
        source_metadata=source_metadata,
    )


def validate_smpl_model_dir(model_dir: Path, model_type: str) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"SMPL model directory does not exist: {model_dir}")
    nested = model_dir / model_type
    if nested.exists():
        return
    if model_dir.name.lower() == model_type and model_dir.is_dir():
        return
    raise FileNotFoundError(
        f"Could not find {model_type.upper()} model files. Expected {nested} or a direct {model_type} directory."
    )
