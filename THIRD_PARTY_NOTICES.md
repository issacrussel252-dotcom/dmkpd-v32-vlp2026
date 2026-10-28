# Third-party notices

The project uses the following external libraries; they are installed separately,
not bundled as binaries in this repository.

| Component | License recorded in the tested installation | Usage |
|---|---|---|
| PyTorch | BSD-3-Clause | Tensor operations, model training and inference |
| torchvision | BSD-3-Clause | ResNet/FPN, Keypoint R-CNN, RoI operations and transforms |
| NumPy | BSD-3-Clause with additional bundled-component notices | Numerical and geometry operations |
| Pillow | MIT-CMU | Image reading and training augmentation |

`src/v9_model.py` extends torchvision's RoIHeads and uses its detection operations.
The torchvision license is retained at `licenses/torchvision-LICENSE`.
Project MIT licensing does not replace any third-party license or grant dataset rights.

The V10 quality head draws conceptual motivation from Mask Scoring R-CNN and
Generalized Focal Loss, but is not a reproduction of either implementation.
The V28 decoder performs bounded local quadratic refinement; it is not a full
implementation of the DARK encoding/decoding pipeline.

No GPL-licensed dependency is directly imported by the released inference code.
These notes identify the direct runtime dependencies, not a legal audit of every
optional package in a user's environment.
