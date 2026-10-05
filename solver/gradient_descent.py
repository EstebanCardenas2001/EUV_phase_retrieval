import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import our Week 1 physics engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.config import OpticsConfig
from physics.simulator import OpticalSystem, twin_phase
from physics.zernike import zernike_polynomial

def total_variation_loss(img: torch.Tensor):
    """
    Calculates the Total Variation (TV) over the last two dims to penalize high-frequency noise.
    """
    # Difference between adjacent rows (vertical edges)
    tv_h = torch.mean(torch.abs(img[..., 1:, :] - img[..., :-1, :]))
    # Difference between adjacent columns (horizontal edges)
    tv_w = torch.mean(torch.abs(img[..., :, 1:] - img[..., :, :-1]))
    return tv_h + tv_w

def solve_phase(simulator: OpticalSystem, target_intensity: torch.Tensor, iterations: int = 300,
                lr: float = 0.1, lambda_tv: float = 0.05, verbose: bool = True):
    """
    Pixelwise gradient-descent phase retrieval fitting all K diversity planes jointly.

    Args:
        target_intensity: Measured stack [..., K, N, N]; leading dims are independent problems.
    Returns:
        (recovered phase [..., N, N], loss history)
    """
    # We start with a completely flat, un-aberrated wavefront (all zeros).
    # requires_grad=True is the magic that tells PyTorch to calculate derivatives for this tensor.
    predicted_phase = torch.zeros(target_intensity.shape[:-3] + target_intensity.shape[-2:],
                                  requires_grad=True, device=target_intensity.device)

    # We use the Adam optimizer. A learning rate of 0.1 is aggressive but works well for phase retrieval.
    # Adam's per-parameter scaling keeps batched problems independent of the batch size.
    optimizer = optim.Adam([predicted_phase], lr=lr)
    target_amplitude = torch.sqrt(target_intensity)
    loss_history = []

    for i in range(iterations):
        optimizer.zero_grad() # Clear old gradients

        # Forward pass: one intensity per diversity plane
        simulated_intensity = simulator(predicted_phase)

        # 1. Data Loss (MSE on Amplitude), averaged over all planes
        loss_data = F.mse_loss(torch.sqrt(simulated_intensity), target_amplitude)

        # 2. Regularization Loss (Physical Smoothness)
        loss_tv = total_variation_loss(predicted_phase)

        # 3. Total Loss
        # lambda_tv is the weight. Too high, and the phase becomes a flat plane.
        # Too low, and the noise remains. 0.05 is a solid baseline for phase retrieval.
        loss = loss_data + lambda_tv * loss_tv

        # Backward pass
        loss.backward()

        # Update the phase tensor
        optimizer.step()

        # Apply the mask to the predicted phase to keep it clean outside the lens
        with torch.no_grad():
            predicted_phase.data *= simulator.mask

        loss_history.append(loss.item())
        if verbose and (i + 1) % 50 == 0:
            print(f"Iteration {i+1}/{iterations} | Loss: {loss.item():.6f}")

    return predicted_phase.detach(), loss_history

def run_inverse_solver(cfg: OpticsConfig = None):
    if cfg is None:
        cfg = OpticsConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device} | diversity planes: {cfg.diversity_defocus} rad RMS")

    # 1. Initialize the Physics Engine
    simulator = OpticalSystem(cfg, device=device)

    # 2. Generate the "Ground Truth" (The hidden reality we want to discover)
    geo = cfg.build_geometry(device)

    # Let's hide a complex combination of aberrations
    true_phase = (zernike_polynomial(geo.rho, geo.theta, simulator.mask, 'astigmatism_vertical') * 0.816 +  # RMS rad (= 2.0 unnormalized)
                  zernike_polynomial(geo.rho, geo.theta, simulator.mask, 'coma_horizontal') * -0.530)     # RMS rad (= -1.5 unnormalized)

    # Run it through the simulator to get the sensor measurement. We detach it from
    # the computation graph because it is our fixed target, not a variable.
    target_intensity = simulator(true_phase, noise_std=0.0).detach()  # [K, N, N]

    print("Starting optimization...")
    recovered_phase, loss_history = solve_phase(simulator, target_intensity)
    return true_phase, target_intensity, recovered_phase, loss_history

def pupil_rmse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """RMS difference inside the pupil over the last two dims."""
    return (a - b)[..., mask > 0.5].pow(2).mean(-1).sqrt()

def wrapped_pupil_rmse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    RMS phase difference inside the pupil, modulo 2*pi and piston: exp(i*a) and exp(i*b) are compared,
    so 2*pi wraps a pixelwise solver can settle into do not count as error.
    """
    d = torch.angle(torch.exp(1j * (a - b)))[..., mask > 0.5]
    piston = torch.angle(torch.exp(1j * d).mean(-1, keepdim=True))
    return torch.angle(torch.exp(1j * (d - piston))).pow(2).mean(-1).sqrt()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classical multi-plane phase retrieval demo")
    parser.add_argument('--diversity', type=str, default=None,
                        help="Comma-separated diversity defocus per plane (default: OpticsConfig)")
    parser.add_argument('--out', type=str, default=None, help="Save the figure here instead of showing it")
    args = parser.parse_args()
    cfg = OpticsConfig() if args.diversity is None else \
        OpticsConfig(diversity_defocus=tuple(float(d) for d in args.diversity.split(',')))
    true_phase, target_intensity, recovered_phase, loss_history = run_inverse_solver(cfg)
    mask = cfg.build_geometry(true_phase.device).mask
    rmse_truth = pupil_rmse(recovered_phase, true_phase, mask).item()
    rmse_twin = pupil_rmse(recovered_phase, twin_phase(true_phase), mask).item()
    print(f"RMSE vs truth: {rmse_truth:.3f} rad | vs twin -phi(-r): {rmse_twin:.3f} rad")

    # 5. Visualize the Results
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    vlim = np.abs(true_phase.cpu().numpy()).max()
    c1 = axes[0].imshow(true_phase.cpu().numpy(), cmap='RdBu', extent=cfg.extent, vmin=-vlim, vmax=vlim)
    axes[0].set_title("Ground Truth Phase (Hidden)")
    fig.colorbar(c1, ax=axes[0])

    # Central region of each diversity plane, tiled left to right
    c = cfg.N // 2
    w = cfg.crop_size // 4
    planes = target_intensity[..., c - w:c + w, c - w:c + w].cpu().numpy() ** 0.5
    axes[1].imshow(np.concatenate(list(planes), axis=1), cmap='inferno')
    axes[1].set_title(f"Sensor Target, defocus {cfg.diversity_defocus}")
    axes[1].axis('off')

    c3 = axes[2].imshow(recovered_phase.cpu().numpy(), cmap='RdBu', extent=cfg.extent, vmin=-vlim, vmax=vlim)
    axes[2].set_title(f"Recovered (RMSE truth {rmse_truth:.2f} / twin {rmse_twin:.2f})")
    fig.colorbar(c3, ax=axes[2])

    axes[3].plot(loss_history, color='blue', linewidth=2)
    axes[3].set_title("Optimization Loss Curve")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("MSE (Amplitude)")
    axes[3].set_yscale('log')
    axes[3].grid(True)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=110, bbox_inches='tight')
        print(f"Saved {args.out}")
    else:
        plt.show()
