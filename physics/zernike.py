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

def noll_to_nm(j: int):
    """
    Converts a Noll index j (>= 1) to the radial order n and signed azimuthal frequency m.
    Modes with even |m| are centrosymmetric (Z(-r) = Z(r)); odd |m| are antisymmetric.
    """
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m

def get_noll_polynomial(j: int, rho: torch.Tensor, theta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Evaluates the Zernike polynomial corresponding to Noll index j (1 to 22).
    """
    if j == 1:
        Z = torch.ones_like(rho)
    elif j == 2:
        Z = 2 * rho * torch.cos(theta)
    elif j == 3:
        Z = 2 * rho * torch.sin(theta)
    elif j == 4:
        Z = 2 * rho**2 - 1
    elif j == 5:
        Z = rho**2 * torch.sin(2 * theta)
    elif j == 6:
        Z = rho**2 * torch.cos(2 * theta)
    elif j == 7:
        Z = (3 * rho**3 - 2 * rho) * torch.sin(theta)
    elif j == 8:
        Z = (3 * rho**3 - 2 * rho) * torch.cos(theta)
    elif j == 9:
        Z = rho**3 * torch.sin(3 * theta)
    elif j == 10:
        Z = rho**3 * torch.cos(3 * theta)
    elif j == 11:
        Z = 6 * rho**4 - 6 * rho**2 + 1
    elif j == 12:
        Z = (4 * rho**4 - 3 * rho**2) * torch.cos(2 * theta)
    elif j == 13:
        Z = (4 * rho**4 - 3 * rho**2) * torch.sin(2 * theta)
    elif j == 14:
        Z = rho**4 * torch.cos(4 * theta)
    elif j == 15:
        Z = rho**4 * torch.sin(4 * theta)
    elif j == 16:
        Z = (10 * rho**5 - 12 * rho**3 + 3 * rho) * torch.cos(theta)
    elif j == 17:
        Z = (10 * rho**5 - 12 * rho**3 + 3 * rho) * torch.sin(theta)
    elif j == 18:
        Z = (5 * rho**5 - 4 * rho**3) * torch.cos(3 * theta)
    elif j == 19:
        Z = (5 * rho**5 - 4 * rho**3) * torch.sin(3 * theta)
    elif j == 20:
        Z = rho**5 * torch.cos(5 * theta)
    elif j == 21:
        Z = rho**5 * torch.sin(5 * theta)
    elif j == 22:
        Z = 20 * rho**6 - 30 * rho**4 + 12 * rho**2 - 1
    else:
        raise ValueError(f"Unsupported Noll index: {j}. Supported indices are 1 to 22.")
    return Z * mask

NOLL_NAME_MAP = {
    'piston': 1,
    'tip': 2,
    'tilt_x': 2,
    'tilt': 3,
    'tilt_y': 3,
    'defocus': 4,
    'astigmatism_oblique': 5,
    'astigmatism_vertical': 6,
    'coma_vertical': 7,
    'coma_horizontal': 8,
    'trefoil_vertical': 9,
    'trefoil_horizontal': 10,
    'spherical': 11,
    'spherical_primary': 11,
    'secondary_astigmatism_vertical': 12,
    'secondary_astigmatism_oblique': 13,
    'quadrafoil_horizontal': 14,
    'quadrafoil_vertical': 15,
    'secondary_coma_horizontal': 16,
    'secondary_coma_vertical': 17,
    'secondary_trefoil_horizontal': 18,
    'secondary_trefoil_vertical': 19,
    'pentafoil_horizontal': 20,
    'pentafoil_vertical': 21,
    'secondary_spherical': 22,
}

def zernike_polynomial(rho: torch.Tensor, theta: torch.Tensor, mask: torch.Tensor, mode):
    """
    Generates a specific Zernike polynomial phase map.
    Accepts string mode names or integer Noll indices (1 to 22).
    """
    if isinstance(mode, int):
        return get_noll_polynomial(mode, rho, theta, mask)
    if isinstance(mode, str) and mode in NOLL_NAME_MAP:
        return get_noll_polynomial(NOLL_NAME_MAP[mode], rho, theta, mask)
    raise ValueError(f"Unknown Zernike mode: {mode}")

def compute_zernike_basis(rho: torch.Tensor, theta: torch.Tensor, mask: torch.Tensor, noll_indices=range(4, 23)) -> torch.Tensor:
    """
    Precomputes a stacked basis tensor of shape [len(noll_indices), H, W].
    """
    return torch.stack([get_noll_polynomial(int(j), rho, theta, mask) for j in noll_indices], dim=0)

if __name__ == "__main__":
    from grid import create_spatial_grid, create_circular_mask
    N_pixels = 256
    L_meters = 0.01  
    pupil_radius = 0.004  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, Y = create_spatial_grid(N=N_pixels, L=L_meters, device=device)
    R = torch.sqrt(X**2 + Y**2)
    mask = create_circular_mask(R, pupil_radius)
    rho, theta = get_polar_coordinates(X, Y, pupil_radius)
    basis = compute_zernike_basis(rho, theta, mask, range(4, 23))
    print(f"Zernike basis precomputed: {basis.shape} on {device}")
