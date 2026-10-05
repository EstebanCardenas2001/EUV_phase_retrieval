# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

EUV phase retrieval in PyTorch: recover a pupil-plane phase map (aberrations expressed in Zernike modes) from a far-field diffraction intensity image. Two approaches share one differentiable physics engine:
- `solver/`: classical per-sample inversion by gradient descent through the simulator.
- `ai/`: a learned network (Attention Res-UNet + Zernike modal head) trained on synthetic data, with Monte Carlo dropout for uncertainty.

## Commands

Environment: Python virtualenv in `.venv/`, dependencies pinned in `requirements.txt` (torch 2.14, CUDA 13 wheels).

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Train (writes saved_models/*.pth and training_progress/epoch_XXX.png relative to CWD)
python ai/train.py --epochs 150 --batch-size 32 --samples-per-epoch 2048 --mode hybrid
python ai/train.py --epochs 2 --samples-per-epoch 64     # quick smoke run

# MC-dropout inference on one fresh synthetic sample -> uq_monte_carlo.png
python ai/inference_uq.py

# Classical gradient-descent solver demo
python solver/gradient_descent.py

# Module self-checks (each file has a __main__ demo)
python ai/unet.py            # shape check + parameter count
python physics/simulator.py  # forward-model visualization
```

Run all scripts from the repo root: output paths (`saved_models/`, `training_progress/`, `uq_monte_carlo.png`) are relative to CWD. There is no test suite, linter, or build step.

`physics/zernike.py`'s `__main__` uses `from grid import ...`, so it only runs from inside `physics/`.

## Architecture

**Imports.** No packaging. Each script does `sys.path.append(<repo root>)` and then imports `physics.*` / `ai.*` absolutely. Only `physics/` has an `__init__.py`. Keep that pattern in new entry-point scripts.

**Shared optical geometry.** Defaults used throughout: `N=256` pixels, `L=0.01` m grid, `pupil_radius=0.004` m. `physics/grid.py` builds the coordinate grid and circular pupil mask. `physics/zernike.py` gives Noll-indexed Zernike polynomials (j=1..22, plus a name→index map) and `compute_zernike_basis`. The models use Noll 4–22 (19 modes; piston/tip/tilt excluded). The dataset, the model and the simulator each rebuild the grid, mask and basis independently, so changing N/L/pupil_radius/noll_indices means keeping all three in sync.

**Forward model** (`physics/simulator.py`, `OpticalSystem`): `mask * exp(i·phase)` → centered `fft2` → `|·|²/N²`, with optional additive Gaussian noise clamped ≥0. It is differentiable, which is what the solver relies on.

**Data** (`ai/dataset.py`): `PhaseRetrievalDataset` synthesizes samples on the fly (random every `__getitem__`, no stored data). Primary modes (4,5,6,7,8,11) are drawn from U(-2,2); the others are sparse U(-0.8,0.8). Piston is removed within the pupil, and the simulator runs with `noise_std=0.02`. Intensity goes through `preprocess_intensity`: center crop to 128, then log10(I+1e-4), then per-sample min-max to [0,1]. Returns `(intensity[1,128,128], phase[1,256,256], coeffs[19])`. Any real-data or inference path must apply the same `preprocess_intensity`.

**Model** (`ai/unet.py`, `AttentionResUNet`, aliased as `UNet`): the input is 128×128 and the output is 256×256.
- A residual encoder (features `[64,128,256]`) feeds a bottleneck with `Dropout2d(0.3)`.
- Modal branch: GAP → MLP (with dropout) → 19 coefficients (clamped ±20) → `DifferentiableZernikeGenerator` (einsum with the registered basis) → 256×256 modal phase.
- Spatial branch: attention-gated U-Net decoder → bilinear upsample to 256 → refinement block → residual phase (clamped ±30).
- `mode`: `'hybrid'` = modal + residual, `'modal'`, or `'unet'`. The output is always multiplied by the pupil mask.
- `forward(..., return_components=True)` → `(total, coeffs, modal, residual)`.
- `enable_mc_dropout()` turns dropout back on after `eval()` for UQ.

**Training** (`ai/train.py`): pure FP32 (no AMP, on purpose), AdamW, ReduceLROnPlateau on the epoch's training loss, grad clipping at 1.0. The loss is masked phase MSE (inside the pupil only) + `coeff_weight` × coefficient MSE + 0.1 × residual-phase energy outside the pupil. There is no validation set. The "best" checkpoint is chosen by training loss, and sample 0 is used as a fixed visual anchor for the per-epoch diagnostic PNGs. The DataLoader uses 4 workers when CUDA is available, and the dataset is built on CPU for worker compatibility.

**Checkpoints.** `saved_models/unet_phase_retrieval_{best,final}.pth` are bare `state_dict`s, with no saved config. `inference_uq.py` builds `UNet()` with default args (hybrid mode, default features), so a checkpoint trained with a different `--mode`/architecture will not load there without matching changes. `saved_models/unet_phase_retrieval.pth` is an older checkpoint and may not match the current architecture.
