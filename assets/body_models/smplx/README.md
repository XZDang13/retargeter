# SMPL-X models

Put the official SMPL-X model files here (`SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`, from
smpl-x.is.tue.mpg.de; licensed, not in git). `tools/export_smplx_model.py` (run by `setup.sh`) turns them into
`smplx_lbs_<gender>.npz`, the plain arrays the retargeter reads for SMPL / SMPL+H / SMPL-X clips
(`--skeleton Configs/skeletons/smplx.json`, DESIGN §17). The exports are gitignored too.
