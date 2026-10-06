# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

EUV phase retrieval in PyTorch: recover a pupil-plane phase map (Zernike aberrations) from far-field diffraction intensity. A single in-focus image cannot determine the phase: φ(r) and its twin −φ(−r) give identical intensity, so the sign of every even-|m| Zernike mode is lost. The repo therefore measures **K phase-diversity planes** (the same wavefront plus known extra defocus, default −1, 0, +1 rad RMS), which breaks the ambiguity. Two approaches share one differentiable physics engine:
- `solver/`: classical pixelwise gradient-descent inversion through the simulator, fitting all K planes jointly.
- `ai/`: Attention Res-UNet + Zernike modal head trained on synthetic K-plane stacks, with MC-dropout and deep-ensemble uncertainty.

## Commands

Environment: Python virtualenv in `.venv/`, dependencies pinned in `requirements.txt` (torch 2.14, CUDA 13 wheels).

```bash
source .venv/bin/activate

# Train -> <save-dir>/{best,final}.pth (best = lowest validation loss), training_log.json, figures in --progress-dir
python ai/train.py --epochs 100 --batch-size 32 --samples-per-epoch 2048 --noise-aug-max 30 --save-dir saved_models/<run> --progress-dir training_progress/<run>
#   --seed S (ensemble members), --scheduler cosine|plateau, --diversity d1,d2,..., --mode hybrid|modal|unet
python ai/train.py --epochs 3 --samples-per-epoch 128 --save-dir /tmp/smoke --progress-dir /tmp/smoke_p   # smoke run

# Evaluation (config is rebuilt from the checkpoint; no geometry flags). Every tool accepts several
# --checkpoint paths, which form a deep ensemble (member-mean prediction, spread over members x MC passes)
python ai/evaluate.py --checkpoint saved_models/<run>/best.pth --out-dir eval/<run>        # per-mode R², twin analysis (--noise-mult N)
python ai/uq_calibration.py --checkpoint saved_models/<run>/best.pth --out-dir eval/<run>  # MC-dropout calibration
python ai/inference_uq.py --checkpoint saved_models/<run>/best.pth                         # -> uq_monte_carlo.png + coefficient table

# Physics studies and self-checks
python ai/diversity_study.py      # diversity layout x defocus sweep with the solver -> eval/diversity_study.png
python ai/crop_energy.py          # energy kept by the intensity crop (worst plane)
python solver/gradient_descent.py # solver demo, prints RMSE vs truth and vs twin (--diversity, --out)
python physics/config.py          # geometry / Q check
python physics/zernike.py         # Zernike RMS + Gram-matrix orthonormality check
python ai/unet.py                 # model shape check
```

Run scripts from the repo root (output paths are relative to CWD). There is no test suite or linter; verification is the self-checks above plus a smoke run. Do not launch full training runs (the user trains on their own GPU) and never write into `saved_models/` root: the legacy `unet_phase_retrieval*.pth` files there are kept on purpose and no longer load.

## Architecture

**Imports.** No packaging. Each script does `sys.path.append(<repo root>)` and imports `physics.*` / `ai.*` / `solver.*` absolutely. Keep that pattern in new entry points.

**Config is the single source of truth** (`physics/config.py`, frozen `OpticsConfig`): `N`, `L`, `pupil_radius`, `noll_indices` (4–22), `crop_size`, `noise_rel`, `noise_aug_max` (training noise augmentation, 1 = off), `diversity_defocus` (K = its length). New fields need defaults so older checkpoints' saved configs still load. `cfg.build_geometry()` is the only place the grid, mask, ρ/θ and Zernike basis are built; the simulator, dataset, model, solver and scripts all take `cfg`. `__post_init__` enforces intensity Nyquist sampling on the actual mask footprint (2D − 1 ≤ N, i.e. Q ≥ 2; default 123 px pupil, Q = 2.08) and `crop_size % 8 == 0`.

**Zernikes** (`physics/zernike.py`) are Noll-normalized (unit RMS on the unit disk), so coefficients are in **RMS radians** and the basis is ~orthonormal on the pixelated pupil. `noll_to_nm` gives (n, m); even |m| ⇔ centrosymmetric ⇔ flips sign under the twin map.

**Forward model** (`physics/simulator.py`): `OpticalSystem(cfg)(phase[..., N, N]) -> intensity[..., K, N, N]`; plane k propagates `mask·exp(i(φ + d_k·Z4))` with a centered FFT over the last two dims, `|·|²/N²`. Noise is additive Gaussian with std `cfg.noise_rel × peak_intensity` (relative, so pupil changes do not change SNR). `twin_phase()` implements φ → −φ(−r); r = 0 sits at pixel N/2, so it is flip + roll by one pixel, not a plain flip.

**Data** (`ai/dataset.py`): samples are synthesized on the fly. `coeff_bound(j)` gives uniform ranges in RMS rad (primary modes 4,5,6,7,8,11 and sparse secondaries). Intensity → `preprocess_intensity`: center crop, log10(I + 1e-4), then **one** min-max across all K planes of a sample (per-channel normalization would erase the relative peak heights that encode defocus blur). Returns `(intensity[K, crop, crop], phase[1, N, N], coeffs[M])`. Any inference path must use the same preprocessing. With `noise_aug_max > 1` each sample's noise is scaled by a log-uniform factor in [1, noise_aug_max]; evaluation code sets `dataset.noise_aug_max = 1.0` to test at fixed noise levels (`coeff_scale` similarly enlarges the coefficient ranges for out-of-distribution tests).

**Model** (`ai/unet.py`, `AttentionResUNet` = `UNet`): input `K × crop × crop`, output `1 × N × N`. Residual encoder → bottleneck (Dropout2d) → modal head (GAP → MLP → coefficients → `DifferentiableZernikeGenerator`) plus attention-gated decoder → residual phase; `mode` = `hybrid` (sum), `modal`, or `unet`; output is masked. Mask/basis buffers are non-persistent: geometry always comes from the config, never from weights.

**Training** (`ai/train.py`): pure FP32 on purpose (no AMP). Loss = pupil-masked phase MSE + `coeff_weight` × coefficient squared error **summed over modes** (same scale as phase MSE thanks to orthonormality) + 0.1 × residual energy outside the pupil. A fixed validation set (512 samples by default, seed 12345, generated without disturbing the training RNG stream) drives best-checkpoint selection. It follows the training distribution, so with `noise_aug_max > 1` each validation sample has its own noise level: validation losses of augmented and fixed-noise runs are not comparable, and evaluation tools test at fixed noise instead (intentional, do not "fix"); the LR follows cosine annealing by default (the plateau scheduler, stepped on the noisy early validation loss, decayed far too early in the first full run). Per-mode validation R² is logged every `--r2-every` epochs.

**Checkpoints** (`ai/checkpoint.py`): `{model_state, config (asdict), model_kwargs, epoch, val_loss, format, train_args}`. Always load through `load_model(path)` / `load_models(paths)` (ensembles; members must share one config), which rebuild config + architecture and load strictly; they raise on missing files and legacy bare state_dicts instead of falling back to random weights or default geometry. `ai/ensemble.py` wraps members behind the single-model `forward` interface.

**Metrics conventions.** `ai/evaluate.py` reports R² both for the modal head and for an LSQ projection of the *total* predicted phase onto the basis, plus twin-aware scores. Reported coefficients are always the projection: the modal head only learns the large primary modes (R² ≈ 0 for Noll 9, 10, 12–21) and the residual branch carries the rest. Pixelwise solver errors must be compared modulo 2π and piston (`wrapped_pupil_rmse`), since the solver can settle into 2π-wrapped but physically identical phases.
