import torch
import math

def get_polar_coordinates(X: torch.Tensor, Y: torch.Tensor, pupil_radius: float):
    """
    Converts Cartesian coordinates to normalized polar coordinates.
    
    Args:
        X, Y (torch.Tensor): The Cartesian coordinate grids.
        pupil_radius (float): The physical radius of the aperture to normalize against.
        
    Returns:
        rho (torch.Tensor): Normalized radial distance (0 to 1 inside the pupil).
        theta (torch.Tensor): Azimuthal angle in radians (-pi to pi).
    """
    R = torch.sqrt(X**2 + Y**2)
    rho = R / pupil_radius
    theta = torch.atan2(Y, X)
    
    return rho, theta

def zernike_polynomial(rho: torch.Tensor, theta: torch.Tensor, mask: torch.Tensor, mode: str):
    """
    Generates a specific Zernike polynomial phase map.
    
    Args:
        rho, theta (torch.Tensor): Normalized polar coordinates.
        mask (torch.Tensor): Binary aperture mask to zero out values outside the pupil.
        mode (str): The name of the aberration to generate.
        
    Returns:
        torch.Tensor: The 2D phase map of the aberration.
    """
    if mode == 'defocus':
        Z = 2 * rho**2 - 1
    elif mode == 'astigmatism_vertical':
        Z = rho**2 * torch.cos(2 * theta)
    elif mode == 'coma_horizontal':
        Z = (3 * rho**3 - 2 * rho) * torch.cos(theta)
    else:
        raise ValueError(f"Unknown Zernike mode: {mode}")
    
    # Multiply by mask to ensure phase is only defined inside the glass lens
    return Z * mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Assuming grid.py is in the same directory, import your previous functions
    from grid import create_spatial_grid, create_circular_mask
    
    N_pixels = 256
    L_meters = 0.01  
    pupil_radius = 0.004  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Generate Cartesian Grid and Mask
    X, Y = create_spatial_grid(N=N_pixels, L=L_meters, device=device)
    R = torch.sqrt(X**2 + Y**2)
    mask = create_circular_mask(R, pupil_radius)
    
    # 2. Convert to Polar
    rho, theta = get_polar_coordinates(X, Y, pupil_radius)
    
    # 3. Generate Aberrations
    phase_defocus = zernike_polynomial(rho, theta, mask, 'defocus')
    phase_astig = zernike_polynomial(rho, theta, mask, 'astigmatism_vertical')
    phase_coma = zernike_polynomial(rho, theta, mask, 'coma_horizontal')

    # 4. Visualize the Phase Maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    c1 = axes[0].imshow(phase_defocus.cpu().numpy(), cmap='RdBu', extent=[-L_meters/2, L_meters/2, -L_meters/2, L_meters/2])
    axes[0].set_title("Defocus")
    fig.colorbar(c1, ax=axes[0], label="Phase (Radians)")
    
    c2 = axes[1].imshow(phase_astig.cpu().numpy(), cmap='RdBu', extent=[-L_meters/2, L_meters/2, -L_meters/2, L_meters/2])
    axes[1].set_title("Astigmatism (Vertical)")
    fig.colorbar(c2, ax=axes[1], label="Phase (Radians)")
    
    c3 = axes[2].imshow(phase_coma.cpu().numpy(), cmap='RdBu', extent=[-L_meters/2, L_meters/2, -L_meters/2, L_meters/2])
    axes[2].set_title("Coma (Horizontal)")
    fig.colorbar(c3, ax=axes[2], label="Phase (Radians)")
    
    plt.tight_layout()
    plt.show()