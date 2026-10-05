import torch
import math
import sys
import os
from dataclasses import dataclass
from typing import NamedTuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.grid import create_spatial_grid, create_circular_mask
from physics.zernike import get_polar_coordinates, compute_zernike_basis

class Geometry(NamedTuple):
    X: torch.Tensor
    Y: torch.Tensor
    mask: torch.Tensor   # [N, N] binary pupil
    rho: torch.Tensor
    theta: torch.Tensor
    basis: torch.Tensor  # [M, N, N] Zernike basis for noll_indices, masked

@dataclass(frozen=True)
class OpticsConfig:
    """
    Single source of truth for the optical geometry and data settings shared by the
    simulator, dataset, model, solver and inference scripts.
    """
    N: int = 256                      # grid pixels per side
    L: float = 0.01                   # physical grid length [m]
    pupil_radius: float = 0.0024      # [m]; 123 px pupil footprint on the default grid -> Q ~ 2.08
    noll_indices: tuple = tuple(range(4, 23))
    crop_size: int = 128              # center crop of the intensity fed to the network
    # Gaussian sensor noise std as a fraction of the unaberrated peak intensity.
    # 1.208e-6 reproduces the previous absolute noise_std=0.02 at pupil_radius=0.004 (peak 16553).
    noise_rel: float = 1.208e-6
    # Training-time noise augmentation: each sample's noise std is noise_rel x a log-uniform factor
    # in [1, noise_aug_max]. 1.0 disables it (fixed SNR, the behaviour of pre-augmentation checkpoints).
    noise_aug_max: float = 1.0
    # Extra Noll-4 defocus per measured plane [rad RMS]. K = 3 at +-1.0 resolves the twin ambiguity:
    # the pixelwise solver recovers 96/96 random training phases (ai/diversity_study.py).
    diversity_defocus: tuple = (-1.0, 0.0, 1.0)

    def __post_init__(self):
        # Normalize sequences to tuples so configs round-trip through asdict/JSON lists
        object.__setattr__(self, 'noll_indices', tuple(int(j) for j in self.noll_indices))
        object.__setattr__(self, 'diversity_defocus', tuple(float(d) for d in self.diversity_defocus))

        if self.noise_aug_max < 1.0:
            raise ValueError(f"noise_aug_max must be >= 1, got {self.noise_aug_max}")
        if self.K < 1:
            raise ValueError("diversity_defocus needs at least one plane")
        if self.N % 2 != 0:
            raise ValueError(f"N must be even, got {self.N}")
        # Intensity is the autocorrelation of the pupil field, so its support is 2D - 1 pixels.
        # It fits on the periodic N-grid without wrap-around only if 2D - 1 <= N, i.e. Q >= ~2.
        if 2 * self.pupil_diameter_px - 1 > self.N:
            raise ValueError(
                f"Intensity undersampled: pupil footprint {self.pupil_diameter_px} px on N={self.N} "
                f"gives Q={self.Q:.3f}; need 2*D - 1 <= N (Q >= 2)."
            )
        # The encoder has three 2x pooling stages
        if self.crop_size % 8 != 0 or not 0 < self.crop_size <= self.N:
            raise ValueError(f"crop_size must be a multiple of 8 in (0, N], got {self.crop_size}")

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def pupil_diameter_px(self) -> int:
        """Pixel footprint of the mask R <= pupil_radius along a central row (offsets -k..k)."""
        return 2 * math.floor(self.pupil_radius / self.dx + 1e-9) + 1

    @property
    def Q(self) -> float:
        return self.N / self.pupil_diameter_px

    @property
    def K(self) -> int:
        """Number of phase-diversity planes (network input channels)."""
        return len(self.diversity_defocus)

    @property
    def num_modes(self) -> int:
        return len(self.noll_indices)

    @property
    def extent(self) -> list:
        """imshow extent of the pupil-plane grid in meters."""
        return [-self.L / 2, self.L / 2, -self.L / 2, self.L / 2]

    def build_geometry(self, device=torch.device('cpu')) -> Geometry:
        X, Y = create_spatial_grid(self.N, self.L, device=device)
        mask = create_circular_mask(torch.sqrt(X**2 + Y**2), self.pupil_radius)
        rho, theta = get_polar_coordinates(X, Y, self.pupil_radius)
        basis = compute_zernike_basis(rho, theta, mask, self.noll_indices)
        return Geometry(X, Y, mask, rho, theta, basis)

if __name__ == "__main__":
    cfg = OpticsConfig()
    geo = cfg.build_geometry()
    row_footprint = int(geo.mask[cfg.N // 2].sum().item())
    assert row_footprint == cfg.pupil_diameter_px, (row_footprint, cfg.pupil_diameter_px)
    print(cfg)
    print(f"dx = {cfg.dx * 1e6:.2f} um | pupil footprint = {cfg.pupil_diameter_px} px | Q = {cfg.Q:.3f}")
    print(f"pupil pixels = {int(geo.mask.sum().item())} | basis {tuple(geo.basis.shape)}")
