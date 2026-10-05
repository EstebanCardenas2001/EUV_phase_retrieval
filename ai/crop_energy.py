import torch
import numpy as np
import argparse
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.dataset import PhaseRetrievalDataset
from physics.config import OpticsConfig

CROPS = (32, 48, 64, 80, 96, 112, 128, 160, 192, 256)

def crop_energy_fractions(intensity: torch.Tensor, crops=CROPS) -> torch.Tensor:
    """Fraction of total intensity inside each centered crop (same crop window as preprocess_intensity)."""
    N = intensity.shape[-1]
    total = intensity.sum(dim=(-2, -1))
    fracs = []
    for c in crops:
        s = (N - c) // 2
        fracs.append(intensity[..., s:s + c, s:s + c].sum(dim=(-2, -1)) / total)
    return torch.stack(fracs, dim=-1)

def analyze(cfg: OpticsConfig, num_samples: int = 2000, seed: int = 0, batch_size: int = 250, target: float = 0.99):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = PhaseRetrievalDataset(cfg, num_samples=num_samples, device=torch.device('cpu'))
    simulator = dataset.simulator.to(device)
    pupil = (dataset.mask > 0.5).to(device)
    # Interior pixels whose right/down neighbours are also in the pupil (for gradient sampling check)
    inner_x = pupil[:, :-1] & pupil[:, 1:]
    inner_y = pupil[:-1, :] & pupil[1:, :]

    fracs, rms, max_grad = [], [], []
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - start)
            phases = torch.stack([dataset.sample_phase()[0] for _ in range(n)]).to(device)
            # Noise-free intensity: the crop decision is about where the signal energy lives
            intensity = simulator(phases)
            # Worst diversity plane per sample: every plane must fit in the crop
            fracs.append(crop_energy_fractions(intensity).amin(dim=1).cpu())
            rms.append(phases[:, pupil].pow(2).mean(1).sqrt().cpu())
            # Phase seen by each plane includes its diversity defocus: [n, K, N, N] -> worst plane
            total = phases.unsqueeze(1) + simulator.diversity_phases
            gx = (total[..., :, 1:] - total[..., :, :-1]).abs()[..., inner_x].amax(-1).amax(-1)
            gy = (total[..., 1:, :] - total[..., :-1, :]).abs()[..., inner_y].amax(-1).amax(-1)
            max_grad.append(torch.maximum(gx, gy).cpu())

    fracs = torch.cat(fracs).numpy()
    rms = torch.cat(rms).numpy()
    max_grad = torch.cat(max_grad).numpy()
    strong = rms >= np.quantile(rms, 0.95)

    print(f"{cfg}\nQ = {cfg.Q:.3f}, pupil footprint {cfg.pupil_diameter_px} px, {num_samples} samples (seed {seed})")
    print(f"Phase RMS in pupil [rad]: median {np.median(rms):.2f}, 95th pct {np.quantile(rms, 0.95):.2f}, max {rms.max():.2f}")
    print(f"Max phase step between adjacent pupil pixels [rad]: median {np.median(max_grad):.3f}, "
          f"max {max_grad.max():.3f} (pupil-plane Nyquist limit: pi = 3.142)")
    print(f"\nEnergy fraction inside centered crop (noise-free intensity, worst of {cfg.K} plane(s))")
    print(f"{'crop':>5} | {'all: mean':>9} {'p1':>7} {'min':>7} | {'top-5% RMS: mean':>16} {'min':>7}")
    for k, c in enumerate(CROPS):
        f, fs = fracs[:, k], fracs[strong, k]
        print(f"{c:>5} | {f.mean():>9.5f} {np.quantile(f, 0.01):>7.4f} {f.min():>7.4f} | {fs.mean():>16.5f} {fs.min():>7.4f}")

    ok = [c for k, c in enumerate(CROPS) if fracs[strong, k].min() >= target and np.quantile(fracs[:, k], 0.01) >= target]
    choice = min(ok) if ok else cfg.N
    print(f"\nSmallest crop with >= {target:.0%} energy for every top-5% sample and the 1st percentile overall: {choice}")
    return choice

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraction of diffraction energy captured by the intensity crop")
    parser.add_argument('--num-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--diversity', type=str, default=None,
                        help="Comma-separated diversity defocus per plane (default: OpticsConfig)")
    args = parser.parse_args()
    cfg = OpticsConfig() if args.diversity is None else \
        OpticsConfig(diversity_defocus=tuple(float(d) for d in args.diversity.split(',')))
    analyze(cfg, num_samples=args.num_samples, seed=args.seed)
