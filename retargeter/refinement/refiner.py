from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from retargeter.newton import RobotSpec, RetargetedMotion, TorchRobotFK, TorchRobotFKResult
from retargeter.preprocess import PreprocessResult
from retargeter.progress import ProgressReporter, get_progress

from .losses import total_refinement_loss


DEFAULT_REFINER_CONFIG = {
    "iterations": 300,
    "lr": 0.01,
    "log_interval": 25,
    "max_root_delta": 0.05,
    "max_joint_delta": 0.25,
    "device": None,
    "dtype": "float32",
    "lbfgs_enabled": False,
    "lbfgs_max_iter": 20,
    "lbfgs_lr": 1.0,
    "lbfgs_line_search_fn": "strong_wolfe",
}


@dataclass
class RefinedMotion:
    fps: float
    robot: str
    joint_names: list[str]
    root_pos_w: np.ndarray
    root_quat_xyzw: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_names: list[str]
    body_pos_w: np.ndarray
    body_quat_xyzw: np.ndarray
    root_delta: np.ndarray
    joint_delta: np.ndarray
    loss_curve: list[dict[str, float | int | str]] = field(default_factory=list)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def validate(self) -> None:
        if self.fps <= 0.0 or not np.isfinite(self.fps):
            raise ValueError(f"fps must be positive and finite, got {self.fps!r}.")
        t = self.num_frames()
        d = len(self.joint_names)
        b = len(self.body_names)
        checks = [
            ("root_pos_w", self.root_pos_w, (t, 3)),
            ("root_quat_xyzw", self.root_quat_xyzw, (t, 4)),
            ("joint_pos", self.joint_pos, (t, d)),
            ("joint_vel", self.joint_vel, (t, d)),
            ("body_pos_w", self.body_pos_w, (t, b, 3)),
            ("body_quat_xyzw", self.body_quat_xyzw, (t, b, 4)),
            ("root_delta", self.root_delta, (t, 3)),
            ("joint_delta", self.joint_delta, (t, d)),
        ]
        for name, value, expected in checks:
            arr = np.asarray(value)
            if arr.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {arr.shape}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or inf values.")
        if len(self.joint_names) != len(set(self.joint_names)):
            raise ValueError("joint_names must be unique.")
        if len(self.body_names) != len(set(self.body_names)):
            raise ValueError("body_names must be unique.")


class TorchMotionRefiner:
    def __init__(
        self,
        robot_spec: RobotSpec,
        torch_fk: TorchRobotFK,
        config: Mapping[str, Any] | None = None,
        log_fn: Callable[[dict[str, float | int | str]], None] | None = None,
        progress: ProgressReporter | None = None,
    ):
        self.robot_spec = robot_spec
        self.torch_fk = torch_fk
        self.config = copy.deepcopy(dict(config or {}))
        self.refiner_config = _refiner_config(self.config)
        self.log_fn = log_fn
        self.progress = get_progress(progress)
        self.device = _torch_device(self.refiner_config["device"], torch_fk)
        self.dtype = _torch_dtype(str(self.refiner_config["dtype"]))

    def refine(self, retargeted: RetargetedMotion, preprocess_result: PreprocessResult) -> RefinedMotion:
        _validate_inputs(retargeted, preprocess_result, self.robot_spec)
        iterations = int(self.refiner_config["iterations"])
        lr = float(self.refiner_config["lr"])
        log_interval = int(self.refiner_config["log_interval"])
        max_root_delta = float(self.refiner_config["max_root_delta"])
        max_joint_delta = float(self.refiner_config["max_joint_delta"])

        retargeted_root = _retargeted_tensor(retargeted.root_pos_w, device=self.device, dtype=self.dtype)
        retargeted_quat = _retargeted_tensor(retargeted.root_quat_xyzw, device=self.device, dtype=self.dtype)
        retargeted_q = _retargeted_tensor(retargeted.joint_pos, device=self.device, dtype=self.dtype)

        raw_root_delta = torch.nn.Parameter(torch.zeros_like(retargeted_root))
        raw_joint_delta = torch.nn.Parameter(torch.zeros_like(retargeted_q))
        optimizer = torch.optim.Adam([raw_root_delta, raw_joint_delta], lr=lr)

        contact_score = _contact_score(preprocess_result)
        ground_height = _ground_height(preprocess_result)
        loss_curve: list[dict[str, float | int | str]] = []

        def evaluate() -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            states = _refined_tensors(
                retargeted_root,
                retargeted_quat,
                retargeted_q,
                raw_root_delta,
                raw_joint_delta,
                max_root_delta=max_root_delta,
                max_joint_delta=max_joint_delta,
            )
            fk_result = self.torch_fk(states["root_pos"], states["root_quat"], states["joint_pos"])
            refined_joint_vel = _joint_velocity(states["joint_pos"], float(retargeted.fps))
            loss, metrics = total_refinement_loss(
                retargeted,
                fk_result,
                states["joint_pos"],
                states["root_pos"],
                refined_joint_vel,
                states["root_delta"],
                states["joint_delta"],
                contact_score,
                ground_height,
                self.robot_spec,
                self.config,
            )
            return loss, metrics, {**states, "fk_result": fk_result, "refined_joint_vel": refined_joint_vel}

        with torch.no_grad():
            initial_loss, initial_metrics, _ = evaluate()
        _record_loss(loss_curve, 0, "adam", initial_loss, initial_metrics, self.log_fn)

        with self.progress.bar(total=iterations, desc="Refine Adam", unit="iter") as bar:
            for iteration in range(1, iterations + 1):
                optimizer.zero_grad()
                loss, metrics, _ = evaluate()
                loss.backward()
                optimizer.step()
                if _should_log(iteration, iterations, log_interval):
                    _record_loss(loss_curve, iteration, "adam", loss, metrics, self.log_fn)
                    bar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4g}"}, refresh=False)
                bar.update(1)

        if bool(self.refiner_config["lbfgs_enabled"]):
            lbfgs = torch.optim.LBFGS(
                [raw_root_delta, raw_joint_delta],
                lr=float(self.refiner_config["lbfgs_lr"]),
                max_iter=int(self.refiner_config["lbfgs_max_iter"]),
                line_search_fn=self.refiner_config["lbfgs_line_search_fn"],
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                closure_loss, _, _ = evaluate()
                closure_loss.backward()
                return closure_loss

            with self.progress.bar(total=1, desc="Refine LBFGS", unit="phase") as bar:
                lbfgs.step(closure)
                bar.update(1)
            with torch.no_grad():
                lbfgs_loss, lbfgs_metrics, _ = evaluate()
            _record_loss(loss_curve, iterations, "lbfgs", lbfgs_loss, lbfgs_metrics, self.log_fn)

        with torch.no_grad():
            final_loss, final_metrics, final_states = evaluate()

        root_pos = _to_numpy(final_states["root_pos"])
        root_quat = _to_numpy(final_states["root_quat"])
        joint_pos = _to_numpy(final_states["joint_pos"])
        root_delta = _to_numpy(final_states["root_delta"])
        joint_delta = _to_numpy(final_states["joint_delta"])
        joint_vel = _to_numpy(_joint_velocity_full(final_states["joint_pos"], float(retargeted.fps)))
        fk_result = final_states["fk_result"]

        quality_metrics = _quality_metrics(
            initial_loss,
            final_loss,
            final_metrics,
            root_delta,
            joint_delta,
            joint_vel,
            contact_available=preprocess_result.contact is not None,
            iteration_count=iterations,
            lbfgs_enabled=bool(self.refiner_config["lbfgs_enabled"]),
        )
        motion = RefinedMotion(
            fps=float(retargeted.fps),
            robot=retargeted.robot,
            joint_names=list(retargeted.joint_names),
            root_pos_w=root_pos,
            root_quat_xyzw=root_quat,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_names=list(fk_result.body_names),
            body_pos_w=_to_numpy(fk_result.body_pos_w),
            body_quat_xyzw=_to_numpy(fk_result.body_quat_xyzw),
            root_delta=root_delta,
            joint_delta=joint_delta,
            loss_curve=loss_curve,
            quality_metrics=quality_metrics,
            metadata={
                "source": "TorchMotionRefiner",
                "retargeted_robot": retargeted.robot,
                "config": copy.deepcopy(self.config),
                "refiner_config": dict(self.refiner_config),
                "ground_height": float(ground_height),
                "contact_available": preprocess_result.contact is not None,
            },
        )
        motion.validate()
        return motion


def run_refinement(
    retargeted: RetargetedMotion,
    preprocess_result: PreprocessResult,
    robot_spec: RobotSpec,
    torch_fk: TorchRobotFK,
    config: Mapping[str, Any] | None = None,
    log_fn: Callable[[dict[str, float | int | str]], None] | None = None,
    progress: ProgressReporter | None = None,
) -> RefinedMotion:
    return TorchMotionRefiner(robot_spec, torch_fk, config=config, log_fn=log_fn, progress=progress).refine(retargeted, preprocess_result)


def run_refinement_batch(
    retargeted: Sequence[RetargetedMotion],
    preprocess_results: Sequence[PreprocessResult],
    robot_spec: RobotSpec,
    torch_fk: TorchRobotFK,
    config: Mapping[str, Any] | None = None,
    log_fn: Callable[[dict[str, float | int | str]], None] | None = None,
    progress: ProgressReporter | None = None,
) -> list[RefinedMotion]:
    motions = list(retargeted)
    preprocess = list(preprocess_results)
    if not motions:
        return []
    if len(motions) != len(preprocess):
        raise ValueError("retargeted and preprocess_results must have the same length.")
    if len(motions) == 1:
        return [run_refinement(motions[0], preprocess[0], robot_spec, torch_fk, config=config, log_fn=log_fn, progress=progress)]
    return BatchedTorchMotionRefiner(robot_spec, torch_fk, config=config, log_fn=log_fn, progress=progress).refine_batch(motions, preprocess)


def _validate_batch_inputs(
    retargeted: Sequence[RetargetedMotion],
    preprocess_results: Sequence[PreprocessResult],
    robot_spec: RobotSpec,
) -> None:
    if len(retargeted) != len(preprocess_results):
        raise ValueError("retargeted and preprocess_results must have the same length.")
    if not retargeted:
        raise ValueError("retargeted batch must be non-empty.")
    joint_names = list(retargeted[0].joint_names)
    body_names = list(retargeted[0].body_names)
    for motion, preprocess_result in zip(retargeted, preprocess_results):
        _validate_inputs(motion, preprocess_result, robot_spec)
        if motion.joint_names != joint_names:
            raise ValueError("All batched RetargetedMotion items must have the same joint_names.")
        if motion.body_names != body_names:
            raise ValueError("All batched RetargetedMotion items must have the same body_names.")


def _padded_tensor(
    arrays: Sequence[np.ndarray],
    max_frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    first = np.asarray(arrays[0])
    output_shape = (len(arrays), int(max_frames)) + tuple(first.shape[1:])
    out = torch.zeros(output_shape, dtype=dtype, device=device)
    for index, value in enumerate(arrays):
        arr = np.asarray(value)
        if arr.shape[1:] != first.shape[1:]:
            raise ValueError(f"Cannot batch arrays with trailing shapes {first.shape[1:]} and {arr.shape[1:]}.")
        out[index, : arr.shape[0]] = torch.as_tensor(arr, dtype=dtype, device=device)
    return out


def _valid_mask(lengths: Sequence[int], max_frames: int, *, device: torch.device) -> torch.Tensor:
    frame_index = torch.arange(int(max_frames), device=device).reshape(1, -1)
    length_tensor = torch.as_tensor([int(length) for length in lengths], dtype=torch.long, device=device).reshape(-1, 1)
    return frame_index < length_tensor


class BatchedTorchMotionRefiner(TorchMotionRefiner):
    def refine_batch(
        self,
        retargeted: Sequence[RetargetedMotion],
        preprocess_results: Sequence[PreprocessResult],
    ) -> list[RefinedMotion]:
        motions = list(retargeted)
        preprocess = list(preprocess_results)
        _validate_batch_inputs(motions, preprocess, self.robot_spec)
        iterations = int(self.refiner_config["iterations"])
        lr = float(self.refiner_config["lr"])
        log_interval = int(self.refiner_config["log_interval"])
        max_root_delta = float(self.refiner_config["max_root_delta"])
        max_joint_delta = float(self.refiner_config["max_joint_delta"])

        lengths = [motion.num_frames() for motion in motions]
        batch_size = len(motions)
        max_frames = max(lengths)
        retargeted_root = _padded_tensor([motion.root_pos_w for motion in motions], max_frames, device=self.device, dtype=self.dtype)
        retargeted_quat = _padded_tensor([motion.root_quat_xyzw for motion in motions], max_frames, device=self.device, dtype=self.dtype)
        retargeted_q = _padded_tensor([motion.joint_pos for motion in motions], max_frames, device=self.device, dtype=self.dtype)
        valid_mask = _valid_mask(lengths, max_frames, device=self.device)

        raw_root_delta = torch.nn.Parameter(torch.zeros_like(retargeted_root))
        raw_joint_delta = torch.nn.Parameter(torch.zeros_like(retargeted_q))
        optimizer = torch.optim.Adam([raw_root_delta, raw_joint_delta], lr=lr)
        contacts = [_contact_score(item) for item in preprocess]
        ground_heights = [_ground_height(item) for item in preprocess]
        loss_curve: list[dict[str, float | int | str]] = []

        def evaluate() -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
            states = _refined_tensors(
                retargeted_root,
                retargeted_quat,
                retargeted_q,
                raw_root_delta,
                raw_joint_delta,
                max_root_delta=max_root_delta,
                max_joint_delta=max_joint_delta,
            )
            flat_root = states["root_pos"].reshape(batch_size * max_frames, 3)
            flat_quat = states["root_quat"].reshape(batch_size * max_frames, 4)
            flat_joint = states["joint_pos"].reshape(batch_size * max_frames, self.robot_spec.num_dofs)
            flat_fk = self.torch_fk(flat_root, flat_quat, flat_joint)
            fk_body_pos = flat_fk.body_pos_w.reshape(batch_size, max_frames, len(flat_fk.body_names), 3)
            fk_body_quat = flat_fk.body_quat_xyzw.reshape(batch_size, max_frames, len(flat_fk.body_names), 4)

            losses: list[torch.Tensor] = []
            metrics_by_key: dict[str, list[torch.Tensor]] = {}
            clip_losses: list[torch.Tensor] = []
            clip_metrics: list[dict[str, torch.Tensor]] = []
            for item_index, motion in enumerate(motions):
                frames = lengths[item_index]
                clip_fk = TorchRobotFKResult(
                    body_names=list(flat_fk.body_names),
                    body_pos_w=fk_body_pos[item_index, :frames],
                    body_quat_xyzw=fk_body_quat[item_index, :frames],
                )
                clip_joint_pos = states["joint_pos"][item_index, :frames]
                clip_root_pos = states["root_pos"][item_index, :frames]
                clip_joint_vel = _joint_velocity(clip_joint_pos, float(motion.fps))
                clip_root_delta = states["root_delta"][item_index, :frames]
                clip_joint_delta = states["joint_delta"][item_index, :frames]
                clip_loss, clip_metric = total_refinement_loss(
                    motion,
                    clip_fk,
                    clip_joint_pos,
                    clip_root_pos,
                    clip_joint_vel,
                    clip_root_delta,
                    clip_joint_delta,
                    contacts[item_index],
                    ground_heights[item_index],
                    self.robot_spec,
                    self.config,
                )
                losses.append(clip_loss)
                clip_losses.append(clip_loss.detach().clone())
                clip_metrics.append({key: value.detach().clone() for key, value in clip_metric.items()})
                for key, value in clip_metric.items():
                    metrics_by_key.setdefault(key, []).append(value)

            total_loss = torch.stack(losses).mean()
            metrics = {key: torch.stack(values).mean().detach().clone() for key, values in metrics_by_key.items()}
            metrics["loss"] = total_loss.detach().clone()
            return total_loss, metrics, {
                **states,
                "fk_body_pos": fk_body_pos,
                "fk_body_quat": fk_body_quat,
                "fk_body_names": list(flat_fk.body_names),
                "clip_losses": clip_losses,
                "clip_metrics": clip_metrics,
                "valid_mask": valid_mask,
            }

        with torch.no_grad():
            initial_loss, initial_metrics, initial_states = evaluate()
        initial_clip_losses = [loss.detach().clone() for loss in initial_states["clip_losses"]]
        _record_loss(loss_curve, 0, "adam", initial_loss, initial_metrics, self.log_fn)

        with self.progress.bar(total=iterations, desc="Batch Refine Adam", unit="iter") as bar:
            for iteration in range(1, iterations + 1):
                optimizer.zero_grad()
                loss, metrics, _ = evaluate()
                loss.backward()
                optimizer.step()
                if _should_log(iteration, iterations, log_interval):
                    _record_loss(loss_curve, iteration, "adam", loss, metrics, self.log_fn)
                    bar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4g}"}, refresh=False)
                bar.update(1)

        if bool(self.refiner_config["lbfgs_enabled"]):
            lbfgs = torch.optim.LBFGS(
                [raw_root_delta, raw_joint_delta],
                lr=float(self.refiner_config["lbfgs_lr"]),
                max_iter=int(self.refiner_config["lbfgs_max_iter"]),
                line_search_fn=self.refiner_config["lbfgs_line_search_fn"],
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                closure_loss, _, _ = evaluate()
                closure_loss.backward()
                return closure_loss

            with self.progress.bar(total=1, desc="Batch Refine LBFGS", unit="phase") as bar:
                lbfgs.step(closure)
                bar.update(1)
            with torch.no_grad():
                lbfgs_loss, lbfgs_metrics, _ = evaluate()
            _record_loss(loss_curve, iterations, "lbfgs", lbfgs_loss, lbfgs_metrics, self.log_fn)

        with torch.no_grad():
            final_loss, final_metrics, final_states = evaluate()

        outputs: list[RefinedMotion] = []
        for item_index, motion in enumerate(motions):
            frames = lengths[item_index]
            root_pos_t = final_states["root_pos"][item_index, :frames]
            root_quat_t = final_states["root_quat"][item_index, :frames]
            joint_pos_t = final_states["joint_pos"][item_index, :frames]
            root_delta_t = final_states["root_delta"][item_index, :frames]
            joint_delta_t = final_states["joint_delta"][item_index, :frames]
            joint_vel_t = _joint_velocity_full(joint_pos_t, float(motion.fps))
            root_delta = _to_numpy(root_delta_t)
            joint_delta = _to_numpy(joint_delta_t)
            joint_vel = _to_numpy(joint_vel_t)
            quality_metrics = _quality_metrics(
                initial_clip_losses[item_index],
                final_states["clip_losses"][item_index],
                final_states["clip_metrics"][item_index],
                root_delta,
                joint_delta,
                joint_vel,
                contact_available=preprocess[item_index].contact is not None,
                iteration_count=iterations,
                lbfgs_enabled=bool(self.refiner_config["lbfgs_enabled"]),
            )
            quality_metrics["batch_size"] = batch_size
            quality_metrics["batch_valid_frame_count"] = int(valid_mask[item_index].sum().detach().cpu())
            item_loss_curve = [dict(record, batch_index=item_index, batch_size=batch_size) for record in loss_curve]
            refined = RefinedMotion(
                fps=float(motion.fps),
                robot=motion.robot,
                joint_names=list(motion.joint_names),
                root_pos_w=_to_numpy(root_pos_t),
                root_quat_xyzw=_to_numpy(root_quat_t),
                joint_pos=_to_numpy(joint_pos_t),
                joint_vel=joint_vel,
                body_names=list(final_states["fk_body_names"]),
                body_pos_w=_to_numpy(final_states["fk_body_pos"][item_index, :frames]),
                body_quat_xyzw=_to_numpy(final_states["fk_body_quat"][item_index, :frames]),
                root_delta=root_delta,
                joint_delta=joint_delta,
                loss_curve=item_loss_curve,
                quality_metrics=quality_metrics,
                metadata={
                    "source": "BatchedTorchMotionRefiner",
                    "retargeted_robot": motion.robot,
                    "config": copy.deepcopy(self.config),
                    "refiner_config": dict(self.refiner_config),
                    "ground_height": float(ground_heights[item_index]),
                    "contact_available": preprocess[item_index].contact is not None,
                    "batch_size": batch_size,
                    "batch_index": item_index,
                },
            )
            refined.validate()
            outputs.append(refined)
        return outputs


def _validate_inputs(retargeted: RetargetedMotion, preprocess_result: PreprocessResult, robot_spec: RobotSpec) -> None:
    retargeted.validate()
    preprocess_result.motion.validate()
    if retargeted.num_frames() != preprocess_result.motion.num_frames():
        raise ValueError(
            f"RetargetedMotion has {retargeted.num_frames()} frames but PreprocessResult motion has "
            f"{preprocess_result.motion.num_frames()}."
        )
    if retargeted.robot != robot_spec.robot:
        raise ValueError(f"RetargetedMotion robot {retargeted.robot!r} does not match RobotSpec {robot_spec.robot!r}.")
    if retargeted.joint_names != robot_spec.actuated_joints:
        raise ValueError("RetargetedMotion joint_names must exactly match RobotSpec actuated_joints.")


def _refiner_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    section = config.get("refiner", {}) if isinstance(config, Mapping) else {}
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise TypeError("config['refiner'] must be a mapping.")
    resolved = dict(DEFAULT_REFINER_CONFIG)
    resolved.update(section)
    _validate_positive_int(resolved["iterations"], "refiner.iterations", allow_zero=True)
    _validate_positive_int(resolved["log_interval"], "refiner.log_interval")
    _validate_positive_float(resolved["lr"], "refiner.lr")
    _validate_positive_float(resolved["max_root_delta"], "refiner.max_root_delta", allow_zero=True)
    _validate_positive_float(resolved["max_joint_delta"], "refiner.max_joint_delta", allow_zero=True)
    _validate_positive_int(resolved["lbfgs_max_iter"], "refiner.lbfgs_max_iter")
    _validate_positive_float(resolved["lbfgs_lr"], "refiner.lbfgs_lr")
    line_search = resolved["lbfgs_line_search_fn"]
    if line_search is not None and line_search != "strong_wolfe":
        raise ValueError("refiner.lbfgs_line_search_fn must be None or 'strong_wolfe'.")
    return resolved


def _validate_positive_int(value, name: str, *, allow_zero: bool = False) -> None:
    int_value = int(value)
    if int_value != value and not isinstance(value, np.integer):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if int_value < 0 or (int_value == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive{' or zero' if allow_zero else ''}, got {value!r}.")


def _validate_positive_float(value, name: str, *, allow_zero: bool = False) -> None:
    float_value = float(value)
    if not np.isfinite(float_value) or float_value < 0.0 or (float_value == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be positive{' or zero' if allow_zero else ''} and finite, got {value!r}.")


def _torch_device(raw_device, torch_fk: TorchRobotFK) -> torch.device:
    if raw_device is not None:
        return torch.device(raw_device)
    try:
        first_param = next(torch_fk.parameters())
        return first_param.device
    except StopIteration:
        pass
    try:
        first_buffer = next(torch_fk.buffers())
        return first_buffer.device
    except StopIteration:
        return torch.device("cpu")


def _torch_dtype(raw_dtype: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "torch.float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "torch.float64": torch.float64,
        "double": torch.float64,
    }
    if raw_dtype not in aliases:
        raise ValueError(f"Unsupported refiner dtype {raw_dtype!r}; expected float32 or float64.")
    return aliases[raw_dtype]


def _retargeted_tensor(value: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def _refined_tensors(
    retargeted_root: torch.Tensor,
    retargeted_quat: torch.Tensor,
    retargeted_q: torch.Tensor,
    raw_root_delta: torch.Tensor,
    raw_joint_delta: torch.Tensor,
    *,
    max_root_delta: float,
    max_joint_delta: float,
) -> dict[str, torch.Tensor]:
    root_delta = float(max_root_delta) * torch.tanh(raw_root_delta)
    joint_delta = float(max_joint_delta) * torch.tanh(raw_joint_delta)
    return {
        "root_delta": root_delta,
        "joint_delta": joint_delta,
        "root_pos": retargeted_root + root_delta,
        "root_quat": retargeted_quat,
        "joint_pos": retargeted_q + joint_delta,
    }


def _joint_velocity(joint_pos: torch.Tensor, fps: float) -> torch.Tensor:
    if joint_pos.shape[0] < 2:
        return joint_pos.new_zeros((0, joint_pos.shape[1]))
    return torch.diff(joint_pos, dim=0) * float(fps)


def _joint_velocity_full(joint_pos: torch.Tensor, fps: float) -> torch.Tensor:
    if joint_pos.shape[0] == 0:
        return joint_pos.clone()
    if joint_pos.shape[0] == 1:
        return torch.zeros_like(joint_pos)
    diff = _joint_velocity(joint_pos, fps)
    return torch.cat((diff[0:1], diff), dim=0)


def _contact_score(preprocess_result: PreprocessResult) -> Mapping[str, np.ndarray]:
    if preprocess_result.contact is None:
        return {}
    return preprocess_result.contact.contact_score


def _ground_height(preprocess_result: PreprocessResult) -> float:
    if preprocess_result.contact is not None:
        return float(preprocess_result.contact.ground_height)
    value = preprocess_result.metadata.get("normalized_ground_height", 0.0)
    if value is None:
        return 0.0
    return float(value)


def _should_log(iteration: int, iterations: int, log_interval: int) -> bool:
    return iteration == iterations or iteration % log_interval == 0


def _record_loss(
    loss_curve: list[dict[str, float | int | str]],
    iteration: int,
    phase: str,
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    log_fn: Callable[[dict[str, float | int | str]], None] | None,
) -> None:
    record: dict[str, float | int | str] = {
        "iteration": int(iteration),
        "phase": phase,
        "loss": float(loss.detach().cpu()),
    }
    for key, value in metrics.items():
        record[key] = float(value.detach().cpu())
    loss_curve.append(record)
    if log_fn is not None:
        log_fn(dict(record))


def _quality_metrics(
    initial_loss: torch.Tensor,
    final_loss: torch.Tensor,
    final_metrics: dict[str, torch.Tensor],
    root_delta: np.ndarray,
    joint_delta: np.ndarray,
    joint_vel: np.ndarray,
    *,
    contact_available: bool,
    iteration_count: int,
    lbfgs_enabled: bool,
) -> dict[str, Any]:
    initial = float(initial_loss.detach().cpu())
    final = float(final_loss.detach().cpu())
    quality: dict[str, Any] = {
        "initial_loss": initial,
        "final_loss": final,
        "loss_improvement": initial - final,
        "max_abs_root_delta": float(np.max(np.abs(root_delta))) if root_delta.size else 0.0,
        "max_abs_joint_delta": float(np.max(np.abs(joint_delta))) if joint_delta.size else 0.0,
        "max_abs_joint_velocity": float(np.max(np.abs(joint_vel))) if joint_vel.size else 0.0,
        "contact_available": bool(contact_available),
        "iteration_count": int(iteration_count),
        "lbfgs_enabled": bool(lbfgs_enabled),
    }
    for key, value in final_metrics.items():
        quality[f"final/{key}"] = float(value.detach().cpu())
    return quality


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()
