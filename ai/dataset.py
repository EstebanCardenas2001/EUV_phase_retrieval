import torch
from torch.utils.data import Dataset
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.config import OpticsConfig
from physics.simulator import OpticalSystem
from physics.zernike import noll_norm

PRIMARY_MODES = (4, 5, 6, 7, 8, 11)

def coeff_bound(j: int) -> float:
    """
    Half-width of the uniform coefficient distribution for Noll mode j, in RMS radians.
    The pre-normalization ranges were U(-2, 2) for primary modes and U(-0.8, 0.8) for the rest,
    on unnormalized polynomials. Dividing by the Noll factor gives the same phase maps,
    so the peak-to-valley distribution is unchanged.
    """
    return (2.0 if j in PRIMARY_MODES else 0.8) / noll_norm(j)

def preprocess_intensity(
    intensity: torch.Tensor,
    crop_size: int = 128,
    epsilon: float = 1e-4
) -> torch.Tensor:
    """
    High-Dynamic-Range (HDR) preprocessing pipeline for EUV diffraction intensity patterns:
    1. Center crops to (crop_size x crop_size) to isolate the diffraction aperture while
       preserving the faint outer rings.
    2. High-Dynamic-Range log10 compression: log10(clamp(I, min=0) + epsilon), boosting
       faint high-angle diffraction fringes that encode high-frequency phase details.
    3. Strict [0, 1] Min-Max normalization per sample, shared across the K diversity planes.
       Every plane carries the same total energy, so the relative peak heights between planes
       encode how much each one is blurred by the diversity defocus; per-channel min-max would erase that.

    Args:
        intensity: Tensor of shape [H, W], [K, H, W], or [B, K, H, W].
        crop_size: Center crop dimension (default: 128).
        epsilon: Numerical stability floor (1e-4).
    Returns:
        Tensor normalized strictly in [0.0, 1.0] with shape matching input layout.
    """
    orig_dim = intensity.dim()
    if orig_dim == 2:
        intensity = intensity.unsqueeze(0)  # [1, H, W]
        
    H, W = intensity.shape[-2], intensity.shape[-1]
    
    # 1. Center crop (preserves faint outer diffraction rings)
    if crop_size is not None and crop_size < H:
        start_y = (H - crop_size) // 2
        end_y = start_y + crop_size
        start_x = (W - crop_size) // 2
        end_x = start_x + crop_size
        cropped = intensity[..., start_y:end_y, start_x:end_x]
    else:
        cropped = intensity
        
    # Clamp non-negative physical intensity
    cropped = torch.clamp(cropped, min=0.0)
    
    # 2. High-Dynamic-Range log10 compression to restore fringe visibility
    i_log = torch.log10(cropped + epsilon)
    
    # 3. Strict [0, 1] Min-Max normalization per sample, shared across planes (i_log is at least [K, H, W] here)
    i_min = i_log.amin(dim=(-3, -2, -1), keepdim=True)
    i_max = i_log.amax(dim=(-3, -2, -1), keepdim=True)
        
    i_norm = (i_log - i_min) / (i_max - i_min + epsilon)
    i_norm = torch.clamp(i_norm, 0.0, 1.0)
    
    if orig_dim == 2:
        i_norm = i_norm.squeeze(0)  # [H, W]
        
    return i_norm

class PhaseRetrievalDataset(Dataset):
    def __init__(
        self,
        cfg: OpticsConfig,
        num_samples: int = 10000,
        device=torch.device('cpu'),
        return_coeffs: bool = True
    ):
        """
        EUV Phase Retrieval Synthetic Dataset.

        Args:
            cfg: Shared optics config (geometry, Zernike modes, crop size, noise level).
            num_samples: Number of phase/intensity pairs per epoch.
            device: Compute device.
            return_coeffs: If True, returns target Zernike coefficient vector.
        """
        self.cfg = cfg
        self.num_samples = num_samples
        self.device = device
        self.noll_indices = cfg.noll_indices
        self.return_coeffs = return_coeffs
        self.crop_size = cfg.crop_size

        self.simulator = OpticalSystem(cfg, device=device)
        self.noise_std = cfg.noise_rel * self.simulator.peak_intensity

        # Precompute the grid and Zernike basis on device for fast synthesis
        geo = cfg.build_geometry(device)
        self.rho, self.theta = geo.rho, geo.theta
        self.mask = self.simulator.mask
        self.basis = geo.basis

    def __len__(self):
        return self.num_samples

    def sample_phase(self):
        """Draws random Zernike coefficients from the training distribution; returns (phase [N, N], coeffs [M])."""
        num_modes = len(self.noll_indices)
        coeffs = torch.zeros(num_modes, dtype=torch.float32, device=self.basis.device)

        # Sample primary aberrations (defocus: 4, astigmatisms: 5,6, comas: 7,8, spherical: 11) with larger magnitude.
        # Coefficients are in RMS radians (Noll-normalized basis); see coeff_bound for the ranges.
        for i, j in enumerate(self.noll_indices):
            if j in PRIMARY_MODES:
                coeffs[i] = random.uniform(-coeff_bound(j), coeff_bound(j))
            else:
                coeffs[i] = random.uniform(-coeff_bound(j), coeff_bound(j)) if random.random() < 0.6 else 0.0
                
        # Synthesize ground truth phase from precomputed basis
        phase = torch.sum(coeffs.view(-1, 1, 1) * self.basis, dim=0)
        
        # 1. Enforce piston removal strictly within the circular pupil aperture:
        # subtract the mean value of the phase strictly within the circular pupil aperture
        # so target phase maps are centered at zero.
        pupil_idx = self.mask > 0.5
        piston = phase[pupil_idx].mean()
        phase = (phase - piston) * self.mask
        return phase, coeffs

    def __getitem__(self, idx):
        phase, coeffs = self.sample_phase()

        with torch.no_grad():
            intensity = self.simulator(phase, noise_std=self.noise_std)  # [K, N, N]

            # 2. High-Dynamic-Range log10 compression + strict [0, 1] Min-Max normalization
            # Applied directly to the cropped intensity planes to preserve faint outer diffraction rings
            intensity_out = preprocess_intensity(
                intensity,
                crop_size=self.crop_size,
                epsilon=1e-4
            )  # [K, crop, crop]

        phase_out = phase.unsqueeze(0)               # [1, N, N]
        
        if self.return_coeffs:
            return intensity_out, phase_out, coeffs
        return intensity_out, phase_out

