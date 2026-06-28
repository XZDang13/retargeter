from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from retargeter.preprocess import REQUIRED_STAGE1_BODY_NAMES, SMPLForwardKinematics, SMPLMotion
import retargeter.preprocess.smpl_fk as smpl_fk_module


class FakeModel:
    num_betas = 10
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    def to(self, device):
        return self

    def __call__(self, **params):
        num_frames = params["transl"].shape[0]
        assert params["expression"].shape[0] == num_frames
        assert params["left_hand_pose"].shape == (num_frames, 45)
        joints = torch.zeros((num_frames, 127, 3), dtype=torch.float32)
        for i in range(127):
            joints[:, i, 0] = float(i)
            joints[:, i, 2] = 0.01 * i
        vertices = torch.zeros((num_frames, 8, 3), dtype=torch.float32)
        return SimpleNamespace(joints=joints, vertices=vertices)


def test_smpl_forward_kinematics_uses_mocked_smplx(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    (model_root / "smplx").mkdir(parents=True)
    monkeypatch.setattr(smpl_fk_module.smplx, "create", lambda *args, **kwargs: FakeModel())

    motion = SMPLMotion(
        model_type="smplx",
        fps=30.0,
        transl=np.zeros((3, 3)),
        global_orient=np.zeros((3, 3)),
        body_pose=np.zeros((3, 63)),
        gender="neutral",
    )

    fk = SMPLForwardKinematics(model_dir=model_root, model_type="smplx", gender="neutral")
    canonical = fk.forward(motion, return_vertices=True)

    assert canonical.body_names == REQUIRED_STAGE1_BODY_NAMES
    assert canonical.body_pos_w.shape == (3, len(REQUIRED_STAGE1_BODY_NAMES), 3)
    assert canonical.body_quat_xyzw.shape == (3, len(REQUIRED_STAGE1_BODY_NAMES), 4)
    assert canonical.vertices_w.shape == (3, 8, 3)
    assert np.array_equal(canonical.mesh_faces, FakeModel.faces)
    assert np.allclose(np.linalg.norm(canonical.body_quat_xyzw, axis=-1), 1.0)
