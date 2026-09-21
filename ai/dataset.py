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
        """
        Procedural dataset generator for optical phase retrieval.
        """
        self.num_samples = num_samples
        self.device = device
        
        # 1. Initialize the physical simulator
        self.simulator = OpticalSystem(N, L, pupil_radius, device=device)
        
        # 2. Pre-compute the spatial grids to save time during data generation
        X, Y = create_spatial_grid(N, L, device=device)
        self.rho, self.theta = get_polar_coordinates(X, Y, pupil_radius)
        self.mask = self.simulator.mask
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        """
        Generates a single random phase/intensity pair.
        PyTorch DataLoaders call this function automatically to build batches.
        """
        # 1. Randomize Zernike coefficients (e.g., between -2.0 and 2.0 radians)
        c_defocus = random.uniform(-2.0, 2.0)
        c_astig = random.uniform(-2.0, 2.0)
        c_coma = random.uniform(-2.0, 2.0)
        
        # 2. Build the complex ground-truth phase map
        phase = (
            zernike_polynomial(self.rho, self.theta, self.mask, 'defocus') * c_defocus +
            zernike_polynomial(self.rho, self.theta, self.mask, 'astigmatism_vertical') * c_astig +
            zernike_polynomial(self.rho, self.theta, self.mask, 'coma_horizontal') * c_coma
        )
        
        # 3. Simulate the sensor measurement
        # We use torch.no_grad() because we are generating static data, not optimizing yet.
        with torch.no_grad():
            intensity = self.simulator(phase, noise_std=0.02)
            
            # Compress dynamic range (matching Week 2 logic)
            intensity = torch.sqrt(intensity)
            
        # 4. Reshape for Convolutional Neural Networks
        # CNNs require [Channels, Height, Width]. We add a channel dimension of 1.
        intensity = intensity.unsqueeze(0)  # Shape: [1, 256, 256]
        phase = phase.unsqueeze(0)          # Shape: [1, 256, 256]
        
        return intensity, phase

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Initialize the dataset and dataloader
    # We generate a small dataset of 100 images for testing
    dataset = PhaseRetrievalDataset(num_samples=100)
    
    # DataLoader handles batching, shuffling, and multi-processing
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Fetch a single batch of 4 image pairs
    batch_intensity, batch_phase = next(iter(dataloader))
    
    print(f"Batch Intensity Tensor Shape: {batch_intensity.shape}")
    print(f"Batch Phase Tensor Shape: {batch_phase.shape}")
    
    # Visualize the first sample in the batch
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # [0] selects the first item in the batch, [0] selects the first (and only) channel
    c1 = axes[0].imshow(batch_phase[0, 0].numpy(), cmap='RdBu', extent=[-0.005, 0.005, -0.005, 0.005])
    axes[0].set_title("U-Net Target: Random Phase")
    fig.colorbar(c1, ax=axes[0])
    
    c2 = axes[1].imshow(batch_intensity[0, 0].numpy(), cmap='inferno')
    axes[1].set_title("U-Net Input: Simulated Intensity")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()