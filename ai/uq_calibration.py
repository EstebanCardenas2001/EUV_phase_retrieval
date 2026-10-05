import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import math
import random
import sys
import os
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.dataset import PhaseRetrievalDataset
from ai.checkpoint import load_models
from ai.evaluate import make_coeff_projector

SIGMA_FLOOR = 1e-6

def mc_collect(models, cfg, dataset, num_samples, mc_passes, batch_size, device, project):
    """
    Predictive distribution on num_samples draws from dataset: the equal-weight mixture over all
    ensemble members x MC-dropout passes (mc_passes = 0: dropout off, one deterministic pass per member).
    Returns per-pupil-pixel and per-coefficient errors of the predictive mean and the predictive std,
    plus per-sample summaries. The predictive mean is piston-removed (piston is unobservable).
    """
    pupil = (dataset.mask > 0.5).to(device)
    for m in models:
        m.eval()
        if mc_passes > 0:
            m.enable_mc_dropout()
    draws = [(m, max(mc_passes, 1)) for m in models]
    total = sum(n for _, n in draws)
    if total < 2:
        raise ValueError("Need at least 2 predictive draws (MC passes or ensemble members) for a std")
    out = {k: [] for k in ("e_pix", "s_pix", "e_coef", "s_coef", "rmse_sample", "sigma_sample")}
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            batch = [dataset[i] for i in range(start, min(start + batch_size, num_samples))]
            x = torch.stack([b[0] for b in batch]).to(device)
            truth = torch.stack([b[1] for b in batch]).squeeze(1).to(device)[:, pupil]  # [B, P]
            true_c = torch.stack([b[2] for b in batch]).to(device)

            s1 = torch.zeros_like(truth)
            s2 = torch.zeros_like(truth)
            coeffs = []
            for model, passes in draws:
                for _ in range(passes):
                    pred = model(x).squeeze(1)                     # [B, N, N], masked
                    coeffs.append(project(pred))
                    pv = pred[:, pupil]
                    pv = pv - pv.mean(1, keepdim=True)
                    s1 += pv
                    s2 += pv ** 2
            mean = s1 / total
            var = (s2 / total - mean ** 2).clamp_min(0) * total / (total - 1)
            coeffs = torch.stack(coeffs)                      # [T, B, M]

            e_pix = mean - truth
            s_pix = var.sqrt()
            out["e_pix"].append(e_pix.flatten().cpu())
            out["s_pix"].append(s_pix.flatten().cpu())
            out["e_coef"].append((coeffs.mean(0) - true_c).cpu())
            out["s_coef"].append(coeffs.std(0, unbiased=True).cpu())
            out["rmse_sample"].append(e_pix.pow(2).mean(1).sqrt().cpu())
            out["sigma_sample"].append(var.mean(1).sqrt().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}

def calibration_metrics(e: np.ndarray, s: np.ndarray, scale: float = 1.0, bins: int = 10) -> dict:
    """Coverage, z-score spread, Gaussian NLL and a sigma-binned reliability curve."""
    e = e.ravel().astype(np.float64)
    s = np.maximum(s.ravel().astype(np.float64) * scale, SIGMA_FLOOR)
    z = e / s
    edges = np.quantile(s, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, bins - 1)
    rel_sigma = [float(np.sqrt(np.mean(s[idx == b] ** 2))) for b in range(bins)]
    rel_rmse = [float(np.sqrt(np.mean(e[idx == b] ** 2))) for b in range(bins)]
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "rms_sigma": float(np.sqrt(np.mean(s ** 2))),
        "coverage_1sigma": float(np.mean(np.abs(z) <= 1)),
        "coverage_2sigma": float(np.mean(np.abs(z) <= 2)),
        "z_rms": float(np.sqrt(np.mean(z ** 2))),
        "nll": float(np.mean(0.5 * np.log(2 * math.pi * s ** 2) + 0.5 * z ** 2)),
        "reliability_sigma": rel_sigma,
        "reliability_rmse": rel_rmse,
    }

def fit_scale(e: np.ndarray, s: np.ndarray) -> float:
    """Post-hoc variance scaling: the single factor that makes the z-scores have unit RMS."""
    s = np.maximum(s.ravel().astype(np.float64), SIGMA_FLOOR)
    return float(np.sqrt(np.mean((e.ravel() / s) ** 2)))

def make_set(cfg, seed, coeff_scale=1.0, noise_mult=None):
    """
    noise_mult=None keeps the training noise distribution (including augmentation), which reproduces
    the training validation set; a number pins every sample to that multiple of the nominal noise.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = PhaseRetrievalDataset(cfg, num_samples=1, device=torch.device('cpu'), return_coeffs=True)
    dataset.coeff_scale = coeff_scale
    if noise_mult is not None:
        dataset.noise_aug_max = 1.0
        dataset.noise_std *= noise_mult
    return dataset

def run(checkpoint, num_samples: int = 1000, ood_samples: int = 500, mc_passes: int = 30,
        batch_size: int = 64, val_seed: int = 12345, val_samples: int = 512, out_dir: str = 'eval'):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    models, cfg, ckpts = load_models(checkpoints, device)
    ckpt = ckpts[0]
    project = make_coeff_projector(cfg, device=device, dtype=torch.float32)
    for path, ck in zip(checkpoints, ckpts):
        print(f"Loaded {path} (epoch {ck['epoch']}, val_loss {ck['val_loss']:.4f})")
    print(f"K={cfg.K} planes {cfg.diversity_defocus}; {len(models)} member(s) x "
          f"{'MC dropout T=' + str(mc_passes) if mc_passes > 0 else 'deterministic (dropout off)'}")

    # Validation set (same seed and stream as training's fixed validation set) is used only to fit the scale
    sets = {
        "val (fit scale)": (make_set(cfg, val_seed), val_samples),
        "test": (make_set(cfg, 1234, noise_mult=1.0), num_samples),
        "OOD coeffs x1.5": (make_set(cfg, 2001, coeff_scale=1.5, noise_mult=1.0), ood_samples),
        "noise x10": (make_set(cfg, 2002, noise_mult=10.0), ood_samples),
        "noise x100": (make_set(cfg, 2003, noise_mult=100.0), ood_samples),
    }
    raw = {name: mc_collect(models, cfg, ds, n, mc_passes, batch_size, device, project)
           for name, (ds, n) in sets.items()}

    val = raw["val (fit scale)"]
    scales = {"pixel": fit_scale(val["e_pix"], val["s_pix"]), "coeff": fit_scale(val["e_coef"], val["s_coef"])}
    print(f"Variance scale fitted on validation: pixel x{scales['pixel']:.2f}, coeff x{scales['coeff']:.2f}\n")

    results = {"checkpoint": checkpoints, "mc_passes": mc_passes, "scales": scales, "sets": {}}
    header = (f"{'set':>16} {'level':>6} | {'RMSE':>6} {'RMS sigma':>9} | {'cov 1s':>6} {'cov 2s':>6} "
              f"{'z RMS':>6} {'NLL':>7} | scaled: {'cov 1s':>6} {'cov 2s':>6} {'NLL':>7}")
    print(header)
    print("-" * len(header))
    for name, r in raw.items():
        entry = {}
        for level, e, s in (("pixel", r["e_pix"], r["s_pix"]), ("coeff", r["e_coef"], r["s_coef"])):
            m_raw = calibration_metrics(e, s)
            m_cal = calibration_metrics(e, s, scale=scales[level])
            entry[level] = {"raw": m_raw, "scaled": m_cal}
            print(f"{name:>16} {level:>6} | {m_raw['rmse']:>6.3f} {m_raw['rms_sigma']:>9.4f} | "
                  f"{m_raw['coverage_1sigma']:>6.1%} {m_raw['coverage_2sigma']:>6.1%} {m_raw['z_rms']:>6.1f} "
                  f"{m_raw['nll']:>7.2f} | {'':>7} {m_cal['coverage_1sigma']:>6.1%} {m_cal['coverage_2sigma']:>6.1%} "
                  f"{m_cal['nll']:>7.2f}")
        rho = spearmanr(r["sigma_sample"], r["rmse_sample"]).statistic
        entry["spearman_sigma_vs_rmse"] = float(rho)
        entry["median_sample_rmse"] = float(np.median(r["rmse_sample"]))
        results["sets"][name] = entry
    print("\nPer-sample ranking (Spearman: RMS predicted sigma vs actual RMSE; 1 = sigma ranks hard samples perfectly)")
    for name, entry in results["sets"].items():
        print(f"  {name:>16}: rho = {entry['spearman_sigma_vs_rmse']:.3f} | median sample RMSE "
              f"{entry['median_sample_rmse']:.3f} rad")
    print("Ideal: coverage 68.3% / 95.4%, z RMS 1.0, reliability on y = x.")

    plot(results, raw["test"], scales, os.path.join(out_dir, 'uq_calibration.png'))
    with open(os.path.join(out_dir, 'uq_calibration.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f"Saved {os.path.join(out_dir, 'uq_calibration.png')} and uq_calibration.json")
    return results

def plot(results, test_raw, scales, path):
    surface, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    colors = {"test": '#2a78d6', "OOD coeffs x1.5": '#eb6834', "noise x10": '#1baf7a', "noise x100": '#eda100'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=surface)

    for ax, level, title in ((axes[0], "pixel", "Pixel reliability"), (axes[1], "coeff", "Coefficient reliability")):
        lo, hi = np.inf, 0
        for name, color in colors.items():
            m = results["sets"][name][level]["raw"]
            ax.plot(m["reliability_sigma"], m["reliability_rmse"], color=color, linewidth=2, marker='o',
                    markersize=5, label=f"{name} (raw)")
            lo = min(lo, min(m["reliability_sigma"]), min(m["reliability_rmse"]))
            hi = max(hi, max(m["reliability_sigma"]), max(m["reliability_rmse"]))
        m = results["sets"]["test"][level]["scaled"]
        ax.plot(m["reliability_sigma"], m["reliability_rmse"], color=colors["test"], linewidth=2, linestyle='--',
                marker='o', markersize=5, label=f"test (scaled x{scales[level]:.1f})")
        lo = max(min(lo, min(m["reliability_sigma"])), 1e-4)
        hi = max(hi, max(m["reliability_sigma"]))
        ax.plot([lo, hi], [lo, hi], color=muted, linewidth=1, linestyle=':', label='ideal (y = x)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('predicted sigma (RMS within bin) [rad]', color=muted)
        ax.set_ylabel('actual RMSE within bin [rad]', color=muted)
        ax.set_title(title, color=ink, loc='left', fontsize=11)

    ax = axes[2]
    z_raw = test_raw["e_pix"] / np.maximum(test_raw["s_pix"], SIGMA_FLOOR)
    z_cal = z_raw / scales["pixel"]
    bins = np.linspace(-6, 6, 61)
    ax.hist(np.clip(z_raw, -6, 6), bins=bins, density=True, color=colors["test"], alpha=0.35,
            label=f"raw ({np.mean(np.abs(z_raw) > 6):.0%} beyond ±6, piled at edges)")
    ax.hist(np.clip(z_cal, -6, 6), bins=bins, density=True, histtype='step', linewidth=2, color=colors["test"],
            label=f"scaled x{scales['pixel']:.1f}")
    xs = np.linspace(-6, 6, 200)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=muted, linestyle=':', linewidth=1.5, label='N(0, 1)')
    ax.set_xlabel('z = error / predicted sigma (test pixels)', color=muted)
    ax.set_ylabel('density', color=muted)
    ax.set_title("z-score distribution", color=ink, loc='left', fontsize=11)

    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=muted, loc='upper left')
    ckpts = results['checkpoint']
    if len(ckpts) == 1:
        run_name = f"{os.path.basename(os.path.dirname(os.path.abspath(ckpts[0])))}/{os.path.basename(ckpts[0])}"
    else:
        run_name = f"ensemble of {len(ckpts)}"
    method = f"MC dropout T={results['mc_passes']}" if results['mc_passes'] > 0 else "dropout off"
    fig.suptitle(f"Uncertainty calibration — {run_name}, {method}",
                 color=ink, x=0.01, ha='left')
    plt.tight_layout()
    plt.savefig(path, dpi=140, facecolor=surface)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration of MC-dropout uncertainty (in- and out-of-distribution)")
    parser.add_argument('--checkpoint', type=str, nargs='+', default=['saved_models/latest/best.pth'],
                        help="One checkpoint, or several ensemble members (same config)")
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--ood-samples', type=int, default=500)
    parser.add_argument('--mc-passes', type=int, default=30,
                        help="MC-dropout passes per member; 0 = dropout off (pure ensemble)")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out-dir', type=str, default='eval')
    args = parser.parse_args()
    run(args.checkpoint, args.num_samples, args.ood_samples, args.mc_passes, args.batch_size, out_dir=args.out_dir)
