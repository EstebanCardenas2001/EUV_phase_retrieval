import torch
import torch.nn as nn
import sys
import os

# Force Python to add the master EUV_phase_retrieval folder to its radar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tools using absolute paths
from physics.config import OpticsConfig
from physics.zernike import zernike_polynomial

def twin_phase(phase: torch.Tensor) -> torch.Tensor:
    """
    Twin-image transform phi(r) -> -phi(-r) on the last two dims. For a centrosymmetric pupil,
    phi and its twin give identical in-focus Fraunhofer intensity.
    The grid has r = 0 at pixel N/2, so r -> -r maps index i to (N - i) mod N,
    which is a flip followed by a roll of one pixel (a plain flip is off by one).
    """
    flipped = torch.flip(phase, dims=(-2, -1))
    return -torch.roll(flipped, shifts=(1, 1), dims=(-2, -1))

class OpticalSystem(nn.Module):
    def __init__(self, cfg: OpticsConfig, device=torch.device('cpu')):
        """
        Initializes the differentiable optical simulator from the shared optics config.
        """
        super().__init__()

        self.cfg = cfg
        self.N = cfg.N

        geo = cfg.build_geometry(device)
        mask = geo.mask

        # Non-persistent buffer: moves with .to(device) and is never updated by the optimizer.
        # The geometry always comes from the config, never from a checkpoint.
        self.register_buffer('mask', mask, persistent=False)

        # Unaberrated on-axis intensity, (pupil pixel count)^2 / N^2. Noise is specified relative to it
        # so changing the pupil size does not silently change the SNR.
        self.peak_intensity = mask.sum().item() ** 2 / self.N ** 2

        # Phase diversity: plane k adds a known defocus d_k * Z4 (Noll-normalized, RMS radians)
        # before propagation. [K, N, N]
        defocus = zernike_polynomial(geo.rho, geo.theta, mask, 4)
        diversity = torch.tensor(cfg.diversity_defocus, dtype=mask.dtype, device=device).view(-1, 1, 1)
        self.register_buffer('diversity_phases', diversity * defocus, persistent=False)

    def forward(self, phase: torch.Tensor, noise_std: float = 0.0):
        """
        Simulates Fraunhofer diffraction from the pupil plane to the sensor plane,
        once per phase-diversity plane in cfg.diversity_defocus.

        Args:
            phase (torch.Tensor): Phase map(s) of shape [..., N, N] (the hidden parameter).
            noise_std (float): Absolute standard deviation of Gaussian sensor noise.
                Use cfg.noise_rel * self.peak_intensity for the configured noise level.

        Returns:
            torch.Tensor: Intensity stack of shape [..., K, N, N].
        """
        # 1. Construct the complex wavefront per plane: U_k = A * mask * exp(i * (phi + d_k * Z4))
        # We assume uniform illumination amplitude (A = 1.0) inside the mask.
        complex_field = self.mask * torch.exp(1j * (phase.unsqueeze(-3) + self.diversity_phases))

        # 2. Fraunhofer Propagation (Differentiable 2D FFT over the last two dims only)
        # ifftshift centers the pupil in the FFT array; fftshift centers the resulting diffraction pattern.
        # The shifts must be restricted to the spatial dims, or they also roll batch/plane dims.
        spatial = (-2, -1)
        field_fft = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(complex_field, dim=spatial), dim=spatial), dim=spatial
        )

        # 3. Sensor Measurement (Intensity = Squared Magnitude)
        # We divide by N to normalize the energy of the FFT
        intensity = (torch.abs(field_fft) / self.N)**2

        # 4. Simulate Sensor Limitations (Noise)
        if noise_std > 0.0:
            noise = torch.randn_like(intensity) * noise_std
            intensity = torch.clamp(intensity + noise, min=0.0)

        return intensity

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = OpticsConfig()

    # Initialize the optical system
    simulator = OpticalSystem(cfg, device=device)

    # Generate a complex aberration (e.g., Coma + Astigmatism)
    geo = cfg.build_geometry(device)

    # Combine Zernike modes with arbitrary severity coefficients
    phase_coma = zernike_polynomial(geo.rho, geo.theta, geo.mask, 'coma_horizontal') * 1.237  # RMS rad (= 3.5 unnormalized)
    phase_astig = zernike_polynomial(geo.rho, geo.theta, geo.mask, 'astigmatism_vertical') * 0.816  # RMS rad (= 2.0 unnormalized)
    total_phase = phase_coma + phase_astig

    # Run the forward pass to get the sensor intensity
    intensity = simulator(total_phase, noise_std=cfg.noise_rel * simulator.peak_intensity)

    # Visualize the Input (Phase) vs Output (Intensity)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    c1 = axes[0].imshow(total_phase.cpu().numpy(), cmap='RdBu', extent=cfg.extent)
    axes[0].set_title("Input: Aberrated Phase Map (Hidden)")
    fig.colorbar(c1, ax=axes[0], label="Radians")

    # We use a logarithmic scale (or power law) for intensity because diffraction
    # central peaks are orders of magnitude brighter than the outer rings.
    # Diversity planes tiled left to right
    c2 = axes[1].imshow(np.concatenate(list(intensity.cpu().numpy()**0.5), axis=1), cmap='inferno')
    axes[1].set_title(f"Output: Sensor Intensity (defocus {cfg.diversity_defocus} rad RMS)")
    axes[1].axis('off') # Hide axes for the "camera" image
    fig.colorbar(c2, ax=axes[1], label="Square Root Intensity")

    plt.tight_layout()
    plt.show()
