import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import random
import sys
import os
from dataclasses import replace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.dataset import PhaseRetrievalDataset
from ai.crop_energy import crop_energy_fractions, CROPS
from physics.config import OpticsConfig
from physics.simulator import OpticalSystem, twin_phase
from solver.gradient_descent import solve_phase, pupil_rmse, wrapped_pupil_rmse

DELTAS = (0.5, 0.75, 1.0, 1.5, 2.0)
LAYOUTS = {
    'K=1 (+d)': lambda d: (d,),
    'K=2 (0, +d)': lambda d: (0.0, d),
    'K=3 (-d, 0, +d)': lambda d: (-d, 0.0, d),
}

def study(num_phases: int = 48, seed: int = 7, iterations: int = 300, out_dir: str = 'eval', deltas=DELTAS):
    """
    Compares phase-diversity layouts and defocus amplitudes d (Noll-4 RMS radians) on the same
    random phases from the training distribution:
      - twin separation: relative L2 distance between amplitude stacks of phi and -phi(-r)
      - classical solver: does pixelwise gradient descent land on the truth or the twin?
      - crop energy: worst-plane energy fraction inside the training crop
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = OpticsConfig(diversity_defocus=(0.0,))

    random.seed(seed)
    torch.manual_seed(seed)
    dataset = PhaseRetrievalDataset(base, num_samples=num_phases)
    phases = torch.stack([dataset.sample_phase()[0] for _ in range(num_phases)]).to(device)
    twins = twin_phase(phases)
    mask = dataset.mask.to(device)
    true_rms = phases[:, mask > 0.5].pow(2).mean(-1).sqrt()
    crop_idx = CROPS.index(base.crop_size)

    configs = [('K=1 in focus', 0.0, (0.0,))]
    configs += [(name, d, layout(d)) for name, layout in LAYOUTS.items() for d in deltas]

    results = []
    print("closer to truth / RMSE<0.1 / wrapped truth use the mod-2pi, piston-free RMSE")
    print(f"{num_phases} phases (seed {seed}), median true RMS {true_rms.median():.2f} rad, solver {iterations} iters\n")
    print(f"{'layout':>16} {'d':>5} | {'twin sep':>8} | {'closer to truth':>15} {'RMSE<0.1':>8} "
          f"{'med RMSE truth':>14} {'med RMSE twin':>13} {'wrapped truth':>13} | {'crop E p1':>9}")
    for name, d, planes in configs:
        cfg = replace(base, diversity_defocus=planes)
        sim = OpticalSystem(cfg, device=device)
        noise_std = cfg.noise_rel * sim.peak_intensity
        with torch.no_grad():
            s_phi = sim(phases)
            s_twin = sim(twins)
            amp_phi, amp_twin = s_phi.sqrt(), s_twin.sqrt()
            sep = ((amp_phi - amp_twin).flatten(1).norm(dim=1) / amp_phi.flatten(1).norm(dim=1))
            crop_e = crop_energy_fractions(s_phi).amin(dim=1)[:, crop_idx]
            target = torch.clamp(s_phi + torch.randn_like(s_phi) * noise_std, min=0.0)

        torch.manual_seed(seed)
        recovered, _ = solve_phase(sim, target, iterations=iterations, verbose=False)
        r_truth = pupil_rmse(recovered, phases, mask)
        r_twin = pupil_rmse(recovered, twins, mask)
        w_truth = wrapped_pupil_rmse(recovered, phases, mask)
        w_twin = wrapped_pupil_rmse(recovered, twins, mask)
        row = {
            'layout': name, 'delta': d, 'planes': planes,
            'twin_sep_median': sep.median().item(),
            # Decided modulo 2*pi: a wrapped solution is physically the same wavefront
            'closer_to_truth': (w_truth < w_twin).float().mean().item(),
            'converged_0p1': (w_truth < 0.1).float().mean().item(),
            'wrapped_rmse_truth_median': w_truth.median().item(),
            'rmse_truth_median': r_truth.median().item(),
            'rmse_twin_median': r_twin.median().item(),
            'crop_energy_p1': float(np.quantile(crop_e.cpu().numpy(), 0.01)),
        }
        results.append(row)
        print(f"{name:>16} {d:>5.2f} | {row['twin_sep_median']:>8.3f} | {row['closer_to_truth']:>15.1%} "
              f"{row['converged_0p1']:>8.1%} {row['rmse_truth_median']:>14.3f} {row['rmse_twin_median']:>13.3f} "
              f"{row['wrapped_rmse_truth_median']:>13.3f} | "
              f"{row['crop_energy_p1']:>9.4f}")

    plot(results, num_phases, os.path.join(out_dir, 'diversity_study.png'))
    with open(os.path.join(out_dir, 'diversity_study.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results

def plot(results, num_phases, path):
    surface, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    colors = {'K=1 (+d)': '#2a78d6', 'K=2 (0, +d)': '#eb6834', 'K=3 (-d, 0, +d)': '#1baf7a'}
    baseline = next(r for r in results if r['layout'] == 'K=1 in focus')
    panels = [('closer_to_truth', 'Solver lands closer to truth than twin', '% of phases', 100),
              ('wrapped_rmse_truth_median', 'Median solver RMSE vs truth (mod 2π)', 'rad', 1),
              ('twin_sep_median', 'Twin separation (amplitude stack)', 'relative L2', 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=surface)
    for ax, (key, title, unit, scale) in zip(axes, panels):
        ax.set_facecolor(surface)
        for name, color in colors.items():
            rows = [r for r in results if r['layout'] == name]
            x = [r['delta'] for r in rows]
            y = [r[key] * scale for r in rows]
            ax.plot(x, y, color=color, linewidth=2, marker='o', markersize=6, label=name)
        ax.axhline(baseline[key] * scale, color=muted, linewidth=1, linestyle='--', label='K=1 in focus')
        ax.set_title(title, color=ink, loc='left', fontsize=11)
        ax.set_xlabel('diversity defocus d [rad RMS, Noll 4]', color=muted)
        ax.set_ylabel(unit, color=muted)
        ax.grid(color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted)
    axes[0].set_ylim(-5, 105)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=muted, loc='lower right')
    fig.suptitle(f"Phase diversity study — {num_phases} random training phases, pixelwise solver",
                 color=ink, x=0.01, ha='left')
    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor=surface)
    plt.close(fig)
    print(f"\nSaved {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose phase-diversity layout and defocus amplitude")
    parser.add_argument('--num-phases', type=int, default=48)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--out-dir', type=str, default='eval')
    parser.add_argument('--deltas', type=str, default=','.join(str(d) for d in DELTAS))
    args = parser.parse_args()
    study(args.num_phases, args.seed, args.iterations, args.out_dir,
          deltas=tuple(float(d) for d in args.deltas.split(',')))
