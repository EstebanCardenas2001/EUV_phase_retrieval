import torch
import torch.nn as nn
import sys
import os

# Force Python to add the master EUV_phase_retrieval folder to its radar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tools using absolute paths
from physics.config import OpticsConfig
from physics.zernike import zernike_polynomial

class OpticalSystem(nn.Module):
    def __init__(self, cfg: OpticsConfig, device=torch.device('cpu')):
        """
        Initializes the differentiable optical simulator from the shared optics config.
        """
        super().__init__()

        self.cfg = cfg
        self.N = cfg.N

        mask = cfg.build_geometry(device).mask

        # Non-persistent buffer: moves with .to(device) and is never updated by the optimizer.
        # The geometry always comes from the config, never from a checkpoint.
        self.register_buffer('mask', mask, persistent=False)

        # Unaberrated on-axis intensity, (pupil pixel count)^2 / N^2. Noise is specified relative to it
        # so changing the pupil size does not silently change the SNR.
        self.peak_intensity = mask.sum().item() ** 2 / self.N ** 2

    def forward(self, phase: torch.Tensor, noise_std: float = 0.0):
        """
        Simulates Fraunhofer diffraction from the pupil plane to the sensor plane.

        Args:
            phase (torch.Tensor): 2D phase map (the hidden parameter).
            noise_std (float): Absolute standard deviation of Gaussian sensor noise.
                Use cfg.noise_rel * self.peak_intensity for the configured noise level.

        Returns:
            torch.Tensor: The 2D intensity measurement at the detector.
        """
        # 1. Construct the complex wavefront: U = A * mask * exp(i * phi)
        # We assume uniform illumination amplitude (A = 1.0) inside the mask.
        complex_field = self.mask * torch.exp(1j * phase)

        # 2. Fraunhofer Propagation (Differentiable 2D FFT)
        # ifftshift centers the pupil in the FFT array; fftshift centers the resulting diffraction pattern.
        field_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(complex_field)))

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
    c2 = axes[1].imshow(intensity.cpu().numpy()**0.5, cmap='inferno')
    axes[1].set_title("Output: Sensor Intensity (Measured)")
    axes[1].axis('off') # Hide axes for the "camera" image
    fig.colorbar(c2, ax=axes[1], label="Square Root Intensity")

    plt.tight_layout()
    plt.show()
