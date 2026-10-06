import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.dataset import PhaseRetrievalDataset
from ai.unet import UNet
from ai.checkpoint import save_checkpoint
from ai.evaluate import r2_score, make_coeff_projector, even_mode_mask
from physics.config import OpticsConfig

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes Mean Squared Error strictly within the valid circular pupil aperture.
    Background pixels outside the pupil mask are completely ignored.
    Properly normalized by the number of valid pixels across the entire batch.
    Indexing valid pixels before computing diff avoids backward pass gradient NaNs.
    """
    valid_mask = (mask > 0.5).expand_as(pred)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    return ((pred_valid - target_valid) ** 2).mean()

def coeff_sse_loss(pred_coeffs: torch.Tensor, true_coeffs: torch.Tensor) -> torch.Tensor:
    """
    Squared coefficient error summed over modes, averaged over the batch. With the Noll-normalized
    (orthonormal) basis this equals the pupil-mean squared phase error of the modal part, so it is on
    the same scale as the phase MSE and coeff_weight = 1 weights the two equally.
    """
    return ((pred_coeffs - true_coeffs) ** 2).sum(dim=1).mean()

def compute_losses(model, intensities, true_phases, true_coeffs, mask, coeff_weight):
    """Forward pass and all loss terms; returns (dict of loss tensors, model outputs)."""
    pred_phases, pred_coeffs, modal_phases, residual_phases = model(intensities, return_components=True)

    loss_phase = masked_mse_loss(pred_phases, true_phases, mask)
    loss_coeff = coeff_sse_loss(pred_coeffs, true_coeffs)

    # Background penalty directly on unmasked residual_phases to prevent background drift
    valid_mask = (mask > 0.5).expand_as(residual_phases)
    bg_loss = (residual_phases[~valid_mask] ** 2).mean()

    total_loss = loss_phase + coeff_weight * loss_coeff + 0.1 * bg_loss
    losses = {"total": total_loss, "phase": loss_phase, "coeff": loss_coeff, "bg": bg_loss}
    return losses, (pred_phases, pred_coeffs)

def make_validation_set(cfg: OpticsConfig, num_samples: int = 512, seed: int = 12345) -> TensorDataset:
    """
    Generates a fixed validation set once, with its own seed. The global RNG state is restored
    afterwards so the validation set does not change the training sample stream.

    The set follows the training distribution, including noise augmentation: with
    cfg.noise_aug_max > 1 each validation sample has its own noise level (log-uniform in
    [1, noise_aug_max] x nominal). Validation losses of augmented runs are therefore not comparable
    with those of fixed-noise runs, nor with evaluate.py / uq_calibration.py, which test at fixed
    noise levels (--noise-mult). This is intentional: checkpoint selection should reflect the noise
    range the model is trained for.
    """
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = PhaseRetrievalDataset(cfg, num_samples=num_samples, device=torch.device('cpu'), return_coeffs=True)
    samples = [dataset[i] for i in range(num_samples)]
    random.setstate(py_state)
    torch.set_rng_state(torch_state)
    return TensorDataset(*[torch.stack(column) for column in zip(*samples)])

def validate(model, loader, mask, coeff_weight, device, project=None):
    """
    Mean validation losses (dropout off). With project given, also returns per-mode R2 of the
    modal-head coefficients and of the LSQ projection of the total predicted phase.
    """
    model.eval()
    sums = {"total": 0.0, "phase": 0.0, "coeff": 0.0, "bg": 0.0}
    true_c, head_c, proj_c = [], [], []
    num_batches = 0
    with torch.no_grad():
        for intensities, true_phases, true_coeffs in loader:
            intensities = intensities.to(device, non_blocking=True)
            true_phases = true_phases.to(device, non_blocking=True)
            true_coeffs = true_coeffs.to(device, non_blocking=True)
            losses, (pred_phases, pred_coeffs) = compute_losses(
                model, intensities, true_phases, true_coeffs, mask, coeff_weight)
            for k in sums:
                sums[k] += losses[k].item()
            num_batches += 1
            if project is not None:
                true_c.append(true_coeffs.cpu())
                head_c.append(pred_coeffs.cpu())
                proj_c.append(project((pred_phases * mask).squeeze(1).cpu()))
    avg = {k: v / num_batches for k, v in sums.items()}
    if project is None:
        return avg, None
    true_c, head_c, proj_c = (torch.cat(x).numpy() for x in (true_c, head_c, proj_c))
    return avg, {"r2_head": r2_score(head_c, true_c), "r2_proj": r2_score(proj_c, true_c)}

def train_model(
    epochs: int = 150,
    batch_size: int = 32,
    samples_per_epoch: int = 2048,
    initial_lr: float = 3e-4,
    coeff_weight: float = 1.0,
    cfg: OpticsConfig = None,
    model_mode: str = 'hybrid',
    save_dir: str = 'saved_models/latest',
    progress_dir: str = 'training_progress',
    val_samples: int = 512,
    val_seed: int = 12345,
    r2_every: int = 10,
    scheduler_name: str = 'cosine',
    min_lr: float = 1e-6,
    plateau_patience: int = 10,
    seed: int = None
):
    if cfg is None:
        cfg = OpticsConfig()
    if seed is not None:
        # Seeds weight init, the training sample stream and (via the torch RNG) DataLoader worker seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    train_args = dict(epochs=epochs, batch_size=batch_size, samples_per_epoch=samples_per_epoch,
                      initial_lr=initial_lr, coeff_weight=coeff_weight, scheduler=scheduler_name,
                      min_lr=min_lr, plateau_patience=plateau_patience, seed=seed,
                      val_samples=val_samples, val_seed=val_seed)
    crop_size = cfg.crop_size
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print("=" * 65)
    print(f"Initializing EUV Phase Retrieval End-to-End Training (Pure FP32) on: {device}")
    if use_cuda:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Total Epochs: {epochs} | Batch Size: {batch_size} | Samples/Epoch: {samples_per_epoch}")
    print(f"Optics: N={cfg.N}, L={cfg.L}, pupil_radius={cfg.pupil_radius} "
          f"(footprint {cfg.pupil_diameter_px} px, Q={cfg.Q:.2f}), noise_rel={cfg.noise_rel:.2e}")
    print(f"Phase diversity: K={cfg.K} planes at defocus {cfg.diversity_defocus} rad RMS")
    print(f"Noise augmentation: {'off' if cfg.noise_aug_max <= 1 else f'log-uniform x1..x{cfg.noise_aug_max:g}'}")
    print(f"Intensity Crop: {crop_size}x{crop_size} (Preserving outer diffraction rings)")
    print(f"Precision: Pure FP32 (Full IEEE-754 precision, no AMP/FP16)")
    print(f"Architecture Mode: {model_mode} | Coeff Loss: sum over modes x {coeff_weight}")
    print(f"Initial Learning Rate: {initial_lr:.2e} | Scheduler: {scheduler_name} (min LR {min_lr:.1e}"
          f"{f', patience {plateau_patience}' if scheduler_name == 'plateau' else ''}) | Seed: {seed}")
    print(f"Validation: {val_samples} fixed samples (seed {val_seed}) | per-mode R2 every {r2_every} epochs")
    print(f"Checkpoints: {save_dir}/best.pth (lowest val loss), {save_dir}/final.pth")
    print("=" * 65)

    # 1. Training Dataset & Fixed Validation Set
    train_dataset = PhaseRetrievalDataset(
        cfg,
        num_samples=samples_per_epoch,
        device=torch.device('cpu'),
        return_coeffs=True
    )

    print("Generating fixed validation set...")
    val_dataset = make_validation_set(cfg, val_samples, val_seed)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=use_cuda)

    # Static visual anchor: validation sample 0, identical across runs with the same val seed
    static_intensity, static_truth, _ = val_dataset[0]

    # Crucial: .copy() prevents memory corruption from PyTorch DataLoader multiprocessing shared memory
    static_mask = train_dataset.mask.cpu().numpy().copy()
    np_intensity = np.concatenate(list(static_intensity.cpu().numpy()), axis=1).copy()  # planes tiled left to right
    np_truth = (static_truth.squeeze().cpu().numpy() * static_mask).copy()

    static_intensity_gpu = static_intensity.unsqueeze(0).to(device)
    static_mask_gpu = train_dataset.mask.unsqueeze(0).unsqueeze(0).to(device)

    # 2. High-Throughput DataLoader for Tesla T4 GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=True if use_cuda else False,
        persistent_workers=True if use_cuda else False,
        prefetch_factor=2 if use_cuda else None
    )

    # 3. Initialize Model and Verify ALL Parameters are Unfrozen (requires_grad = True)
    model_kwargs = dict(in_channels=cfg.K, out_channels=1, mode=model_mode, negative_slope=0.1)
    model = UNet(cfg, **model_kwargs).to(device)
    model_kwargs["features"] = list(model.features)

    # Ensure every single layer has requires_grad = True
    for name, param in model.named_parameters():
        param.requires_grad = True

    frozen_params = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen_params:
        raise RuntimeError(f"Unexpected frozen parameters found: {frozen_params}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Verified all model parameters are trainable: {trainable_params:,} / {total_params:,} (requires_grad = True).")
    print(f"Modal head parameters: {sum(p.numel() for p in model.zernike_head.parameters()):,} (trainable: True)")

    # 4. Optimizer, Scheduler and R2 projector in Pure FP32.
    # Cosine annealing (default) is independent of the noisy early validation loss, which made the
    # plateau scheduler halve the LR far too early; plateau is kept with a longer patience and an LR floor.
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                         patience=plateau_patience, min_lr=min_lr)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    project = make_coeff_projector(cfg)
    is_even = even_mode_mask(cfg.noll_indices)

    # 5. Pure FP32 End-to-End Training Loop
    best_val = float('inf')
    history = []

    for epoch in range(epochs):
        current_epoch = epoch + 1
        model.train()

        # Guard: verify requires_grad remains True for all parameters
        for p in model.parameters():
            if not p.requires_grad:
                p.requires_grad = True

        sums = {"total": 0.0, "phase": 0.0, "coeff": 0.0, "bg": 0.0}
        num_batches = len(train_loader)

        for batch_idx, (intensities, true_phases, true_coeffs) in enumerate(train_loader):
            intensities = intensities.to(device, non_blocking=True)
            true_phases = true_phases.to(device, non_blocking=True)
            true_coeffs = true_coeffs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Pure FP32 forward pass - no autocast
            losses, _ = compute_losses(model, intensities, true_phases, true_coeffs, static_mask_gpu, coeff_weight)

            # Pure FP32 backward pass - no scaler
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in sums:
                sums[k] += losses[k].item()

        train_avg = {k: v / num_batches for k, v in sums.items()}

        # Validation (dropout off); per-mode R2 every r2_every epochs and at the end
        log_r2 = current_epoch % r2_every == 0 or current_epoch == epochs
        val_avg, r2 = validate(model, val_loader, static_mask_gpu, coeff_weight, device,
                               project if log_r2 else None)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"==> Epoch {current_epoch:03d}/{epochs} | "
            f"Train: {train_avg['total']:.4f} (phase {train_avg['phase']:.4f}, coeff {train_avg['coeff']:.4f}, "
            f"bg {train_avg['bg']:.4f}) | "
            f"Val: {val_avg['total']:.4f} (phase {val_avg['phase']:.4f}, coeff {val_avg['coeff']:.4f}) | "
            f"LR: {current_lr:.2e}"
        )
        record = {"epoch": current_epoch, "lr": current_lr, "train": train_avg, "val": val_avg}
        if r2 is not None:
            r2p = r2["r2_proj"]
            worst = int(np.argmin(r2p))
            print(f"    Val R2 (projected): odd {r2p[~is_even].mean():.3f} | even {r2p[is_even].mean():.3f} | "
                  f"worst Noll {cfg.noll_indices[worst]} = {r2p[worst]:.3f} | "
                  f"head mean {r2['r2_head'].mean():.3f}")
            print("    per mode: " + " ".join(f"{j}:{v:.2f}" for j, v in zip(cfg.noll_indices, r2p)))
            record["val_r2_proj"] = r2p.tolist()
            record["val_r2_head"] = r2["r2_head"].tolist()
        history.append(record)

        if scheduler_name == 'plateau':
            scheduler.step(val_avg["total"])
        else:
            scheduler.step()

        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            save_checkpoint(os.path.join(save_dir, 'best.pth'), model, cfg, model_kwargs,
                            current_epoch, best_val, train_args=train_args)

        with open(os.path.join(save_dir, 'training_log.json'), 'w') as f:
            json.dump({"noll": list(cfg.noll_indices), "coeff_weight": coeff_weight, "history": history}, f, indent=1)

        # 6. Visual Epoch Tracking: 5-panel diagnostic figure
        if epochs <= 10 or current_epoch % 10 == 0 or current_epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_p, _, modal_p, res_p = model(static_intensity_gpu, return_components=True)
                pred_p = torch.nan_to_num(pred_p, nan=0.0, posinf=0.0, neginf=0.0)
                modal_p = torch.nan_to_num(modal_p, nan=0.0, posinf=0.0, neginf=0.0)
                res_p = torch.nan_to_num(res_p, nan=0.0, posinf=0.0, neginf=0.0)

                pred_np = pred_p.squeeze().cpu().numpy() * static_mask
                modal_np = modal_p.squeeze().cpu().numpy() * static_mask
                res_np = res_p.squeeze().cpu().numpy() * static_mask

                pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=0.0, neginf=0.0)
                modal_np = np.nan_to_num(modal_np, nan=0.0, posinf=0.0, neginf=0.0)
                res_np = np.nan_to_num(res_np, nan=0.0, posinf=0.0, neginf=0.0)

            fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

            axes[0].imshow(np_intensity, cmap='inferno')
            axes[0].set_title(f"Input planes {cfg.diversity_defocus} ({crop_size}x{crop_size})", fontsize=11)
            axes[0].axis('off')

            c2 = axes[1].imshow(np_truth, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
            axes[1].set_title("Target True Phase (val #0)", fontsize=11)
            fig.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)

            c3 = axes[2].imshow(modal_np, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
            axes[2].set_title(f"Modal Baseline (Ep {current_epoch})", fontsize=11)
            fig.colorbar(c3, ax=axes[2], fraction=0.046, pad=0.04)

            c4 = axes[3].imshow(pred_np, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
            axes[3].set_title(f"Total Prediction (Ep {current_epoch})", fontsize=11)
            fig.colorbar(c4, ax=axes[3], fraction=0.046, pad=0.04)

            err_map = np.abs(pred_np - np_truth) * static_mask
            err_map = np.nan_to_num(err_map, nan=0.0, posinf=0.0, neginf=0.0)
            # Scale to the data (99th percentile, at least 0.1 rad) so small late-training errors stay visible
            pupil_err = err_map[static_mask > 0.5]
            err_vmax = max(float(np.percentile(pupil_err, 99)), 0.1)
            c5 = axes[4].imshow(err_map, cmap='magma', extent=cfg.extent, vmin=0.0, vmax=err_vmax)
            axes[4].set_title(f"Abs Error (Ep {current_epoch}, RMS {np.sqrt(np.mean(pupil_err ** 2)):.3f} rad)",
                              fontsize=11)
            fig.colorbar(c5, ax=axes[4], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(os.path.join(progress_dir, f'epoch_{current_epoch:03d}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

    save_checkpoint(os.path.join(save_dir, 'final.pth'), model, cfg, model_kwargs, epochs, val_avg["total"],
                    train_args=train_args)
    print(f"\nPure FP32 training complete. Best validation loss: {best_val:.6f} -> {save_dir}/best.pth")
    return best_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EUV Phase Retrieval End-to-End Pure FP32 Training on Tesla T4")
    parser.add_argument('--epochs', type=int, default=150, help="Total epochs (default: 150)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for T4 in pure FP32 (default: 32)")
    parser.add_argument('--samples-per-epoch', type=int, default=2048, help="Samples per epoch (default: 2048)")
    parser.add_argument('--crop-size', type=int, default=OpticsConfig.crop_size,
                        help=f"Crop size for intensity (default: {OpticsConfig.crop_size})")
    parser.add_argument('--diversity', type=str, default=','.join(str(d) for d in OpticsConfig.diversity_defocus),
                        help="Comma-separated diversity defocus per plane, Noll-4 RMS radians (sets K)")
    parser.add_argument('--save-dir', type=str, default='saved_models/latest', help="Checkpoint output directory")
    parser.add_argument('--progress-dir', type=str, default='training_progress', help="Diagnostic figure directory")
    parser.add_argument('--lr', type=float, default=3e-4, help="Initial learning rate (default: 3e-4)")
    parser.add_argument('--coeff-weight', type=float, default=1.0,
                        help="Weight of the coefficient loss (summed over modes, same scale as phase MSE)")
    parser.add_argument('--mode', type=str, default='hybrid', choices=['hybrid', 'modal', 'unet'], help="Architecture mode")
    parser.add_argument('--val-samples', type=int, default=512, help="Fixed validation set size (default: 512)")
    parser.add_argument('--val-seed', type=int, default=12345, help="Validation set seed")
    parser.add_argument('--r2-every', type=int, default=10, help="Log per-mode validation R2 every N epochs")
    parser.add_argument('--noise-aug-max', type=float, default=OpticsConfig.noise_aug_max,
                        help="Per-sample noise multiplier drawn log-uniformly from [1, this]; 1 disables (default)")
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau'])
    parser.add_argument('--min-lr', type=float, default=1e-6, help="LR floor (cosine eta_min / plateau min_lr)")
    parser.add_argument('--plateau-patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None, help="Seed for init and training data (ensemble members)")

    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        initial_lr=args.lr,
        coeff_weight=args.coeff_weight,
        cfg=OpticsConfig(crop_size=args.crop_size,
                         diversity_defocus=tuple(float(d) for d in args.diversity.split(',')),
                         noise_aug_max=args.noise_aug_max),
        model_mode=args.mode,
        save_dir=args.save_dir,
        progress_dir=args.progress_dir,
        val_samples=args.val_samples,
        val_seed=args.val_seed,
        r2_every=args.r2_every,
        scheduler_name=args.scheduler,
        min_lr=args.min_lr,
        plateau_patience=args.plateau_patience,
        seed=args.seed
    )
