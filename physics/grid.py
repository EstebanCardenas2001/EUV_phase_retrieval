import torch

def create_spatial_grid(N: int, L: float, device: torch.device = torch.device('cpu')):
    """
    Generates a 2D spatial coordinate grid centered at zero.
    
    Args:
        N (int): Number of pixels along one dimension (must be power of 2, e.g., 256, 512).
        L (float): Total physical length of the grid in meters (e.g., 0.01 for 10 mm).
        device (torch.device): The device (CPU/GPU) where the tensors will reside.
        
    Returns:
        X, Y (torch.Tensor): 2D tensors representing the x and y physical coordinates.
    """
    if N % 2 != 0:
        raise ValueError("Grid resolution N must be an even number (power of 2 preferred for FFT).")

    # 1. Calculate pixel pitch
    dx = L / N

    # 2. Create a 1D vector of physical coordinates
    # We shift by half a pixel pitch to ensure the physical center is perfectly symmetrical
    # and to align with how FFTshift handles zero-frequency components.
    x_1d = torch.linspace(-L/2, L/2 - dx, N, device=device)
    
    # 3. Expand the 1D vector into a 2D meshgrid
    # indexing='ij' ensures matrix-style indexing (rows, columns)
    Y, X = torch.meshgrid(x_1d, x_1d, indexing='ij')
    
    return X, Y

def create_circular_mask(R: torch.Tensor, radius: float):
    """
    Creates a binary circular aperture mask.
    
    Args:
        R (torch.Tensor): 2D tensor of radial distances.
        radius (float): The physical radius of the aperture in meters.
        
    Returns:
        torch.Tensor: A 2D binary mask (1.0 inside radius, 0.0 outside).
    """
    # Create a boolean tensor (True if inside radius) and convert to floats (1.0 or 0.0)
    mask = (R <= radius).float()
    return mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_pixels = 256
    L_meters = 0.01  # 10 mm total grid size
    pupil_radius = 0.004  # 4 mm lens radius
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Generate the spatial grid
    X, Y = create_spatial_grid(N=N_pixels, L=L_meters, device=device)
    
    # 2. Calculate radial distances
    R = torch.sqrt(X**2 + Y**2)
    
    # 3. Generate the circular aperture mask
    mask = create_circular_mask(R, pupil_radius)
    
    # Convert to NumPy for Matplotlib
    R_np = R.cpu().numpy()
    mask_np = mask.cpu().numpy()

    # 4. Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot Radial distance
    c1 = axes[0].imshow(R_np, cmap='viridis', extent=[-L_meters/2, L_meters/2, -L_meters/2, L_meters/2])
    axes[0].set_title("Radial Distance (R)")
    fig.colorbar(c1, ax=axes[0], label="Meters")
    
    # Plot the resulting Aperture Mask
    c2 = axes[1].imshow(mask_np, cmap='gray', extent=[-L_meters/2, L_meters/2, -L_meters/2, L_meters/2])
    axes[1].set_title(f"Aperture Mask (Radius = {pupil_radius*1000} mm)")
    fig.colorbar(c2, ax=axes[1], label="Amplitude Transmission")
    
    plt.tight_layout()
    plt.show()