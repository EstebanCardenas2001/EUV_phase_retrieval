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

def noll_norm(j: int) -> float:
    """Noll normalization factor giving unit RMS over the unit disk: sqrt(n+1) if m = 0, else sqrt(2(n+1))."""
    n, m = noll_to_nm(j)
    return math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))

def get_noll_polynomial(j: int, rho: torch.Tensor, theta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Evaluates the Noll-normalized Zernike polynomial for Noll index j (1 to 22).
    Each mode has unit RMS over the unit disk, so coefficients are in RMS radians.
    """
    if j == 1:
        Z = torch.ones_like(rho)
    elif j == 2:
        Z = rho * torch.cos(theta)
    elif j == 3:
        Z = rho * torch.sin(theta)
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
    return noll_norm(j) * Z * mask

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
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physics.config import OpticsConfig
    cfg = OpticsConfig()
    geo = cfg.build_geometry()
    pupil = geo.mask > 0.5
    print(f"Pixelated pupil: {cfg.pupil_diameter_px} px footprint, {int(pupil.sum())} pixels")

    # Normalization check over all Noll modes 1..22 on the pixelated pupil.
    # Gram matrix G_ij = mean over pupil of Z_i * Z_j; orthonormal modes give G = I.
    js = list(range(1, 23))
    full = compute_zernike_basis(geo.rho, geo.theta, geo.mask, js)[:, pupil].double()  # [22, P]
    G = full @ full.T / full.shape[1]
    rms = G.diag().sqrt()
    off = G - torch.diag(G.diag())
    i, k = divmod(int(off.abs().argmax()), len(js))
    print(f"{'Noll':>4} {'(n,m)':>7} {'norm':>6} {'RMS':>7}")
    for idx, j in enumerate(js):
        n, m = noll_to_nm(j)
        print(f"{j:>4} {f'({n},{m})':>7} {noll_norm(j):>6.3f} {rms[idx].item():>7.4f}")
    print(f"RMS range: [{rms.min().item():.4f}, {rms.max().item():.4f}]")
    print(f"Max |off-diagonal| Gram entry: {off.abs().max().item():.4f} (Noll {js[i]} x Noll {js[k]})")
    sub = G[3:, 3:]  # Noll 4..22, the modes the models use
    print(f"Noll 4-22 only: max |off-diagonal| {(sub - torch.diag(sub.diag())).abs().max().item():.4f}")
    assert (rms - 1).abs().max() < 0.05, "Mode RMS deviates from 1 by more than 5%"
    assert off.abs().max() < 0.05, "Basis is not close to orthonormal on the pixelated pupil"
