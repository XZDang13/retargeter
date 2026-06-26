from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from .canonical import CanonicalHumanMotion, REQUIRED_STAGE1_BODY_NAMES
from .config import PreprocessConfig
from .contact import FootContactEstimator, FootContactResult
from .ground import GroundEstimate, GroundPlaneEstimator
from .lowpass import MotionLowPassFilter


@dataclass
class PreprocessResult:
    motion: CanonicalHumanMotion
    ground: GroundEstimate | None
    contact: FootContactResult | None
    warnings: list[str]
    metadata: dict = field(default_factory=dict)


class MotionPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config

    def process(self, motion: CanonicalHumanMotion) -> PreprocessResult:
        motion.validate(required_bodies=REQUIRED_STAGE1_BODY_NAMES)
        collected_warnings: list[str] = []

        processed = motion.copy()
        lowpass_applied = False
        if self.config.lowpass.enabled:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                processed = MotionLowPassFilter(self.config.lowpass).apply(processed)
            collected_warnings.extend(str(w.message) for w in caught)
            lowpass_applied = True

        ground_estimate: GroundEstimate | None = None
        original_ground_height: float | None = None
        normalized_ground_height: float | None = None
        if self.config.ground.enabled:
            ground_estimator = GroundPlaneEstimator(self.config.ground)
            ground_estimate = ground_estimator.estimate(processed)
            original_ground_height = ground_estimate.ground_height
            processed = ground_estimator.normalize_to_ground(processed, ground_estimate.ground_height)
            normalized_ground_height = 0.0

        contact_result: FootContactResult | None = None
        if self.config.contact.enabled:
            contact_result = FootContactEstimator(self.config.contact).estimate(
                processed,
                normalized_ground_height if normalized_ground_height is not None else 0.0,
            )

        contact_ratio = {}
        if contact_result is not None:
            contact_ratio = {
                name: float(values.mean()) for name, values in contact_result.contact_binary.items()
            }

        metadata = {
            "lowpass_applied": lowpass_applied,
            "original_ground_height": original_ground_height,
            "normalized_ground_height": normalized_ground_height,
            "ground_confidence": None if ground_estimate is None else ground_estimate.confidence,
            "contact_available": contact_result is not None,
            "contact_ratio": contact_ratio,
        }
        processed.metadata.update(metadata)

        return PreprocessResult(
            motion=processed,
            ground=ground_estimate,
            contact=contact_result,
            warnings=collected_warnings,
            metadata=metadata,
        )

