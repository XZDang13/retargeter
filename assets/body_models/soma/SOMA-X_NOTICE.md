# SOMA-X / MHR model data

`soma_x_model.npz` holds arrays exported from [SOMA-X](https://github.com/NVlabs/SOMA-X) at commit
`366ed27cf88d88d9702c4f9f30cea046c0cc391c` by `tools/export_soma_x_model.py`: the low-LOD SOMA rig, SOMA's skeleton
transfer and topology transfer data, and the LOD6 MHR identity model it uses.

- SOMA-X: Copyright NVIDIA Corporation, licensed under the Apache License, Version 2.0.
- MHR (Momentum Human Rig) model assets distributed with SOMA-X: licensed under the Apache License, Version 2.0.

A copy of the license is at <https://www.apache.org/licenses/LICENSE-2.0>. The arrays are a subset of those assets,
converted to NumPy and, for the Laplacian, its precomputed system matrix. They contain no BONES-SEED data.
