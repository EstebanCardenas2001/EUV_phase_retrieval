import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
import random

# Add the parent directory to the path so we can import the physics engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.simulator import OpticalSystem
from physics.grid import create_spatial_grid
from physics.zernike import get_polar_coordinates, zernike_polynomial

class PhaseRetrievalDataset(Dataset):
    def __init__(self, num_samples: int = 10000, N: int = 256, L: float = 0.01, pupil_radius: float = 0.004, device=torch.device('cpu')):
        self.num_samples = num_samples
        self.device = device
        
        self.simulator = OpticalSystem(N, L, pupil_radius, device=device)
        
        X, Y = create_spatial_grid(N, L, device=device)
        self.rho, self.theta = get_polar_coordinates(X, Y, pupil_radius)
        self.mask = self.simulator.mask
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        c_defocus = random.uniform(-2.0, 2.0)
        c_astig = random.uniform(-2.0, 2.0)
        c_coma = random.uniform(-2.0, 2.0)
        
        phase = (
            zernike_polynomial(self.rho, self.theta, self.mask, 'defocus') * c_defocus +
            zernike_polynomial(self.rho, self.theta, self.mask, 'astigmatism_vertical') * c_astig +
            zernike_polynomial(self.rho, self.theta, self.mask, 'coma_horizontal') * c_coma
        )
        
        with torch.no_grad():
            intensity = self.simulator(phase, noise_std=0.02)
            
            # Logarithmic compression to boost faint interference fringes
            epsilon = 1e-9
            intensity_log = torch.log10(intensity + epsilon)
            
            # Min-Max Normalization to bound values between [0, 1] for the CNN
            i_min = intensity_log.min()
            i_max = intensity_log.max()
            intensity = (intensity_log - i_min) / (i_max - i_min + epsilon)
            
        intensity = intensity.unsqueeze(0)
        phase = phase.unsqueeze(0)
        
        return intensity, phase