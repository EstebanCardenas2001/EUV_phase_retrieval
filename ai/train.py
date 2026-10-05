import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.dataset import PhaseRetrievalDataset
from ai.unet import UNet
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

def train_model(
    epochs: int = 150,
    batch_size: int = 32,
    samples_per_epoch: int = 2048,
    initial_lr: float = 3e-4,
    coeff_weight: float = 1.0,
    cfg: OpticsConfig = None,
    model_mode: str = 'hybrid',
    save_dir: str = 'saved_models',
    progress_dir: str = 'training_progress'
):
    if cfg is None:
        cfg = OpticsConfig()
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
    print(f"Intensity Crop: {crop_size}x{crop_size} (Preserving outer diffraction rings)")
    print(f"Precision: Pure FP32 (Full IEEE-754 precision, no AMP/FP16)")
    print(f"Architecture Mode: {model_mode} | Coeff Loss Weight: {coeff_weight}")
    print(f"Initial Learning Rate: {initial_lr:.2e}")
    print("=" * 65)

    # 1. Dataset & Static Validation Sample
    train_dataset = PhaseRetrievalDataset(
        cfg,
        num_samples=samples_per_epoch,
        device=torch.device('cpu'),
        return_coeffs=True
    )

    print("Generating static validation anchor...")
    static_sample = train_dataset[0]
    static_intensity = static_sample[0]
    static_truth = static_sample[1]
    
    # Crucial: .copy() prevents memory corruption from PyTorch DataLoader multiprocessing shared memory
    static_mask = train_dataset.mask.cpu().numpy().copy()
    np_intensity = static_intensity.squeeze().cpu().numpy().copy()
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
    model = UNet(cfg, in_channels=1, out_channels=1, mode=model_mode, negative_slope=0.1).to(device)
    
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

    # 4. Optimizer, Scheduler, and Losses in Pure FP32
    coeff_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    # 5. Pure FP32 End-to-End Training Loop
    best_loss = float('inf')

    for epoch in range(epochs):
        current_epoch = epoch + 1
        model.train()

        # Guard: verify requires_grad remains True for all parameters
        for p in model.parameters():
            if not p.requires_grad:
                p.requires_grad = True

        epoch_loss = 0.0
        epoch_phase_loss = 0.0
        epoch_coeff_loss = 0.0
        epoch_bg_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (intensities, true_phases, true_coeffs) in enumerate(train_loader):
            intensities = intensities.to(device, non_blocking=True)
            true_phases = true_phases.to(device, non_blocking=True)
            true_coeffs = true_coeffs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Pure FP32 forward pass - no autocast
            pred_phases, pred_coeffs, modal_phases, residual_phases = model(intensities, return_components=True)

            loss_phase = masked_mse_loss(pred_phases, true_phases, static_mask_gpu)
            loss_coeff = coeff_criterion(pred_coeffs, true_coeffs)

            # Background penalty directly on unmasked residual_phases to prevent background drift
            valid_mask = (static_mask_gpu > 0.5).expand_as(residual_phases)
            bg_loss = (residual_phases[~valid_mask] ** 2).mean()

            total_loss = loss_phase + coeff_weight * loss_coeff + 0.1 * bg_loss

            # Pure FP32 backward pass - no scaler
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_phase_loss += loss_phase.item()
            epoch_coeff_loss += loss_coeff.item()
            epoch_bg_loss += bg_loss.item()

        avg_loss = epoch_loss / num_batches
        avg_phase_loss = epoch_phase_loss / num_batches
        avg_coeff_loss = epoch_coeff_loss / num_batches
        avg_bg_loss = epoch_bg_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"==> Epoch {current_epoch:03d}/{epochs} [FP32 End-to-End] | "
            f"Total: {avg_loss:.6f} | "
            f"Phase MSE: {avg_phase_loss:.6f} | "
            f"Coeff MSE: {avg_coeff_loss:.6f} | "
            f"BG Loss: {avg_bg_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'unet_phase_retrieval_best.pth'))

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
            axes[0].set_title(f"Input Sensor ({crop_size}x{crop_size})", fontsize=11)
            axes[0].axis('off')

            c2 = axes[1].imshow(np_truth, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
            axes[1].set_title("Target True Phase", fontsize=11)
            fig.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)

            c3 = axes[2].imshow(modal_np, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
            axes[2].set_title(f"Modal Baseline (Ep {current_epoch})", fontsize=11)
            fig.colorbar(c3, ax=axes[2], fraction=0.046, pad=0.04)

            c4 = axes[3].imshow(pred_np, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
            axes[3].set_title(f"Total Prediction (Ep {current_epoch})", fontsize=11)
            fig.colorbar(c4, ax=axes[3], fraction=0.046, pad=0.04)

            err_map = np.abs(pred_np - np_truth) * static_mask
            err_map = np.nan_to_num(err_map, nan=0.0, posinf=0.0, neginf=0.0)
            c5 = axes[4].imshow(err_map, cmap='magma', extent=cfg.extent, vmin=0.0, vmax=4.0)
            axes[4].set_title(f"Abs Error Map (Ep {current_epoch})", fontsize=11)
            fig.colorbar(c5, ax=axes[4], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(os.path.join(progress_dir, f'epoch_{current_epoch:03d}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

    torch.save(model.state_dict(), os.path.join(save_dir, 'unet_phase_retrieval_final.pth'))
    print(f"\nPure FP32 training complete. Best model saved with loss: {best_loss:.6f}")
    return best_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EUV Phase Retrieval End-to-End Pure FP32 Training on Tesla T4")
    parser.add_argument('--epochs', type=int, default=150, help="Total epochs (default: 150)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for T4 in pure FP32 (default: 32)")
    parser.add_argument('--samples-per-epoch', type=int, default=2048, help="Samples per epoch (default: 2048)")
    parser.add_argument('--crop-size', type=int, default=OpticsConfig.crop_size,
                        help=f"Crop size for intensity (default: {OpticsConfig.crop_size})")
    parser.add_argument('--save-dir', type=str, default='saved_models', help="Checkpoint output directory")
    parser.add_argument('--progress-dir', type=str, default='training_progress', help="Diagnostic figure directory")
    parser.add_argument('--lr', type=float, default=3e-4, help="Initial learning rate (default: 3e-4)")
    parser.add_argument('--coeff-weight', type=float, default=1.0, help="Weight for auxiliary Zernike coefficient loss")
    parser.add_argument('--mode', type=str, default='hybrid', choices=['hybrid', 'modal', 'unet'], help="Architecture mode")
    
    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        initial_lr=args.lr,
        coeff_weight=args.coeff_weight,
        cfg=OpticsConfig(crop_size=args.crop_size),
        model_mode=args.mode,
        save_dir=args.save_dir,
        progress_dir=args.progress_dir
    )

