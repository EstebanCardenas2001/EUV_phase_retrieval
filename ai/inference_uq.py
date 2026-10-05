import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Force Python to add the root folder to its radar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from ai.dataset import PhaseRetrievalDataset
from ai.checkpoint import load_model
from ai.evaluate import make_coeff_projector

def run_monte_carlo_inference(model_path: str, mc_passes: int = 50):
    # 1. Hardware Optimization: Utilize CUDA if available, fallback to MPS/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Running Monte Carlo Inference on: {device}")

    # 2. Rebuild the trained network from its checkpoint (geometry and diversity planes included).
    # Raises if the checkpoint is missing or does not match; never runs with random weights
    model, cfg, _ = load_model(model_path, device)
    print(f"Loaded trained model weights from {model_path} (K={cfg.K} planes {cfg.diversity_defocus}).")

    # 3. Initialize the dataset to generate a single unseen test sample
    test_dataset = PhaseRetrievalDataset(cfg, num_samples=1, device=device, return_coeffs=True)
    intensity_input, true_phase, true_coeffs = test_dataset[0]

    # Add the batch dimension [1, K, crop, crop] expected by the network
    intensity_input = intensity_input.unsqueeze(0).to(device)
    true_phase = true_phase.to(device)

    # 4. Setup Monte Carlo Dropout
    model.eval()
    model.enable_mc_dropout()

    print(f"Executing {mc_passes} forward passes for Uncertainty Quantification...")

    # Reported coefficients are the LSQ projection of the total predicted phase onto the basis:
    # in hybrid mode the residual branch carries real signal that the modal head misses.
    project = make_coeff_projector(cfg, device=device, dtype=torch.float32)
    predictions, coeffs = [], []

    with torch.no_grad():
        for _ in range(mc_passes):
            pred = model(intensity_input)
            predictions.append(pred)
            coeffs.append(project(pred.squeeze(1))[0])

    # 5. Statistical Aggregation
    predictions_tensor = torch.stack(predictions)
    coeffs = torch.stack(coeffs)  # [T, M]
    return {
        "intensity": intensity_input[0],
        "truth": true_phase.squeeze(),
        "mean": predictions_tensor.mean(dim=0).squeeze(),
        "std": predictions_tensor.std(dim=0).squeeze(),
        "mask": test_dataset.mask.squeeze(),
        "true_coeffs": true_coeffs.cpu(),
        "coeff_mean": coeffs.mean(0).cpu(),
        "coeff_std": coeffs.std(0).cpu(),
        "cfg": cfg,
    }

def print_coefficient_table(r):
    cfg = r["cfg"]
    print(f"\nZernike coefficients [rad RMS] (projected total phase, MC mean +- std)")
    print(f"{'Noll':>4} {'true':>7} {'pred':>7} {'std':>6} {'z':>6}")
    for j, t, m, s in zip(cfg.noll_indices, r["true_coeffs"].tolist(), r["coeff_mean"].tolist(), r["coeff_std"].tolist()):
        print(f"{j:>4} {t:>7.3f} {m:>7.3f} {s:>6.3f} {(m - t) / max(s, 1e-6):>6.1f}")
    err = r["coeff_mean"] - r["true_coeffs"]
    print(f"coefficient RMSE {err.pow(2).mean().sqrt():.4f} rad | within 2 std: "
          f"{(err.abs() <= 2 * r['coeff_std']).float().mean():.0%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monte Carlo dropout inference on one fresh synthetic sample")
    parser.add_argument('--checkpoint', type=str, default='saved_models/latest/best.pth')
    parser.add_argument('--mc-passes', type=int, default=50)
    parser.add_argument('--out', type=str, default='uq_monte_carlo.png')
    args = parser.parse_args()

    r = run_monte_carlo_inference(args.checkpoint, mc_passes=args.mc_passes)
    cfg = r["cfg"]
    print_coefficient_table(r)

    # Move tensors to CPU and convert to NumPy for Matplotlib
    intensity = np.concatenate(list(r["intensity"].cpu().numpy()), axis=1)  # diversity planes tiled left to right
    mask = r["mask"].cpu().numpy()
    truth = r["truth"].cpu().numpy()
    # Apply the physical lens mask to the AI predictions
    mean_pred = r["mean"].cpu().numpy() * mask
    uncertainty = r["std"].cpu().numpy() * mask

    # 6. Visualize the Portfolio Deliverable
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), gridspec_kw={'width_ratios': [1.3, 1, 1, 1, 1.4]})

    axes[0].imshow(intensity, cmap='inferno')
    axes[0].set_title(f"Input: Sensor Intensity, defocus {cfg.diversity_defocus}")
    axes[0].axis('off')

    c2 = axes[1].imshow(truth, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
    axes[1].set_title("Target: True Phase Map")
    fig.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)

    c3 = axes[2].imshow(mean_pred, cmap='RdBu', extent=cfg.extent, vmin=-2.0, vmax=2.0)
    axes[2].set_title("Output: Predicted Phase (Mean)")
    fig.colorbar(c3, ax=axes[2], fraction=0.046, pad=0.04)

    c4 = axes[3].imshow(uncertainty, cmap='magma', extent=cfg.extent)
    axes[3].set_title("UQ: Predictive Std [rad]")
    fig.colorbar(c4, ax=axes[3], fraction=0.046, pad=0.04)

    x = np.arange(cfg.num_modes)
    axes[4].bar(x - 0.2, r["true_coeffs"].numpy(), width=0.4, color='#9a9994', label='true')
    axes[4].bar(x + 0.2, r["coeff_mean"].numpy(), width=0.4, color='#2a78d6',
                yerr=2 * r["coeff_std"].numpy(), ecolor='#0b0b0b', capsize=2, label='predicted ± 2 std')
    axes[4].axhline(0, color='#52514e', linewidth=0.8)
    axes[4].set_xticks(x, [str(j) for j in cfg.noll_indices], fontsize=8)
    axes[4].set_xlabel('Noll index')
    axes[4].set_ylabel('coefficient [rad RMS]')
    axes[4].set_title("Zernike Coefficients (projected)")
    axes[4].legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Inference UQ visual saved to {args.out}")
