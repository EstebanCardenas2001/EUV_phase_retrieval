import torch
import torch.nn as nn
import sys
import os

# Force Python to add the master EUV_phase_retrieval folder to its radar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tools using absolute paths
from physics.grid import create_spatial_grid, create_circular_mask
from physics.zernike import get_polar_coordinates, zernike_polynomial

class OpticalSystem(nn.Module):
    def __init__(self, N: int = 256, L: float = 0.01, pupil_radius: float = 0.004, device=torch.device('cpu')):
        """
        Initializes the differentiable optical simulator.
        """
        super().__init__()
        
        self.N = N
        self.L = L
        self.pupil_radius = pupil_radius
        
        # 1. Generate the static grids
        X, Y = create_spatial_grid(N, L, device=device)
        R = torch.sqrt(X**2 + Y**2)
        
        # 2. Generate the static aperture mask
        mask = create_circular_mask(R, pupil_radius)
        
        # register_buffer ensures these tensors are saved as part of the module's state,
        # are moved automatically when you call model.to(device), but are NOT updated by the optimizer.
        self.register_buffer('mask', mask)
        
    def forward(self, phase: torch.Tensor, noise_std: float = 0.0):
        """
        Simulates Fraunhofer diffraction from the pupil plane to the sensor plane.
        
        Args:
            phase (torch.Tensor): 2D phase map (the hidden parameter).
            noise_std (float): Standard deviation of Gaussian sensor noise.
            
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
    
    # Initialize the optical system
    simulator = OpticalSystem(N=256, L=0.01, pupil_radius=0.004, device=device)
    
    # Generate a complex aberration (e.g., Coma + Astigmatism)
    X, Y = create_spatial_grid(256, 0.01, device=device)
    rho, theta = get_polar_coordinates(X, Y, 0.004)
    
    # Combine Zernike modes with arbitrary severity coefficients
    phase_coma = zernike_polynomial(rho, theta, simulator.mask, 'coma_horizontal') * 3.5
    phase_astig = zernike_polynomial(rho, theta, simulator.mask, 'astigmatism_vertical') * 2.0
    total_phase = phase_coma + phase_astig
    
    # Run the forward pass to get the sensor intensity
    intensity = simulator(total_phase, noise_std=0.01)
    
    # Visualize the Input (Phase) vs Output (Intensity)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    c1 = axes[0].imshow(total_phase.cpu().numpy(), cmap='RdBu', extent=[-0.005, 0.005, -0.005, 0.005])
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