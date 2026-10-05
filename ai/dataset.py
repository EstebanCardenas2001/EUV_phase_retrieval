import torch
from torch.utils.data import Dataset
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.simulator import OpticalSystem
from physics.grid import create_spatial_grid
from physics.zernike import get_polar_coordinates, compute_zernike_basis

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
    3. Strict [0, 1] Min-Max normalization per sample to bound tensor inputs for the CNN.
    
    Args:
        intensity: Tensor of shape [H, W], [1, H, W], or [B, 1, H, W].
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
    
    # 3. Strict [0, 1] Min-Max normalization per sample
    if i_log.dim() >= 3:
        i_min = i_log.amin(dim=(-2, -1), keepdim=True)
        i_max = i_log.amax(dim=(-2, -1), keepdim=True)
    else:
        i_min = i_log.min()
        i_max = i_log.max()
        
    i_norm = (i_log - i_min) / (i_max - i_min + epsilon)
    i_norm = torch.clamp(i_norm, 0.0, 1.0)
    
    if orig_dim == 2:
        i_norm = i_norm.squeeze(0)  # [H, W]
        
    return i_norm

class PhaseRetrievalDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 10000,
        N: int = 256,
        L: float = 0.01,
        pupil_radius: float = 0.004,
        device=torch.device('cpu'),
        noll_indices=tuple(range(4, 23)),
        return_coeffs: bool = True,
        crop_size: int = 128
    ):
        """
        EUV Phase Retrieval Synthetic Dataset.
        
        Args:
            num_samples: Number of phase/intensity pairs per epoch.
            N: Grid dimension (256x256).
            L: Physical grid length in meters.
            pupil_radius: Pupil aperture radius in meters.
            device: Compute device.
            noll_indices: Primary Zernike modes to synthesize (Noll 4 to 22).
            return_coeffs: If True, returns target Zernike coefficient vector.
            crop_size: Center crop dimension for intensity input (128x128 preserves faint outer rings).
        """
        self.num_samples = num_samples
        self.N = N
        self.L = L
        self.pupil_radius = pupil_radius
        self.device = device
        self.noll_indices = tuple(noll_indices)
        self.return_coeffs = return_coeffs
        self.crop_size = crop_size
        
        self.simulator = OpticalSystem(N, L, pupil_radius, device=device)
        
        X, Y = create_spatial_grid(N, L, device=device)
        self.rho, self.theta = get_polar_coordinates(X, Y, pupil_radius)
        self.mask = self.simulator.mask
        
        # Precompute the Zernike basis on device for fast synthesis
        self.basis = compute_zernike_basis(self.rho, self.theta, self.mask, self.noll_indices).to(device)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        num_modes = len(self.noll_indices)
        coeffs = torch.zeros(num_modes, dtype=torch.float32, device=self.basis.device)
        
        # Sample primary aberrations (defocus: 4, astigmatisms: 5,6, comas: 7,8, spherical: 11) with larger magnitude
        for i, j in enumerate(self.noll_indices):
            if j in (4, 5, 6, 7, 8, 11):
                coeffs[i] = random.uniform(-2.0, 2.0)
            else:
                coeffs[i] = random.uniform(-0.8, 0.8) if random.random() < 0.6 else 0.0
                
        # Synthesize ground truth phase from precomputed basis
        phase = torch.sum(coeffs.view(-1, 1, 1) * self.basis, dim=0)
        
        # 1. Enforce piston removal strictly within the circular pupil aperture:
        # subtract the mean value of the phase strictly within the circular pupil aperture
        # so target phase maps are centered at zero.
        pupil_idx = self.mask > 0.5
        piston = phase[pupil_idx].mean()
        phase = (phase - piston) * self.mask
        
        with torch.no_grad():
            intensity = self.simulator(phase, noise_std=0.02)
            
            # 2. High-Dynamic-Range log10 compression + strict [0, 1] Min-Max normalization
            # Applied directly to 128x128 cropped intensity to preserve faint outer diffraction rings
            intensity_norm = preprocess_intensity(
                intensity,
                crop_size=self.crop_size,
                epsilon=1e-4
            )
            
        intensity_out = intensity_norm.unsqueeze(0) if intensity_norm.dim() == 2 else intensity_norm  # [1, 128, 128]
        phase_out = phase.unsqueeze(0)               # [1, 256, 256]
        
        if self.return_coeffs:
            return intensity_out, phase_out, coeffs
        return intensity_out, phase_out

