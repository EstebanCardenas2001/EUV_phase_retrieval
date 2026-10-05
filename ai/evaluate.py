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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.dataset import PhaseRetrievalDataset
from ai.unet import UNet
from physics.config import OpticsConfig
from physics.zernike import noll_to_nm, get_noll_polynomial

def twin_phase(phase: torch.Tensor) -> torch.Tensor:
    """
    Twin-image transform phi(r) -> -phi(-r) on the last two dims.
    The grid has r = 0 at pixel N/2, so r -> -r maps index i to (N - i) mod N,
    which is a flip followed by a roll of one pixel (a plain flip is off by one).
    """
    flipped = torch.flip(phase, dims=(-2, -1))
    return -torch.roll(flipped, shifts=(1, 1), dims=(-2, -1))

def load_checkpoint_strict(model: torch.nn.Module, path: str, device) -> None:
    """Loads weights or raises. Never falls back to random weights."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Checkpoint {path} does not match the model architecture:\n{e}") from e

def r2_score(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-column coefficient of determination over samples."""
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot

def self_check(dataset: PhaseRetrievalDataset, is_even: np.ndarray) -> dict:
    """
    Verifies the twin map before trusting the analysis:
    1. twin(phase) equals the phase synthesized with even-mode coefficients negated.
    2. A phase and its twin give identical noise-free Fraunhofer intensity.
    """
    coeffs = torch.randn(len(dataset.noll_indices))
    phase = torch.sum(coeffs.view(-1, 1, 1) * dataset.basis, dim=0)
    sign = torch.where(torch.from_numpy(is_even), -1.0, 1.0)
    phase_twin_coeffs = torch.sum((coeffs * sign).view(-1, 1, 1) * dataset.basis, dim=0)
    coeff_err = (twin_phase(phase) - phase_twin_coeffs).abs().max().item()

    with torch.no_grad():
        i_phi = dataset.simulator(phase, noise_std=0.0)
        i_twin = dataset.simulator(twin_phase(phase), noise_std=0.0)
    intensity_rel_err = ((i_phi - i_twin).abs().max() / i_phi.max()).item()

    assert coeff_err < 1e-4, f"Twin map does not match even-mode sign flip (max err {coeff_err:.2e})"
    assert intensity_rel_err < 1e-4, f"Twin intensity differs (max rel err {intensity_rel_err:.2e})"
    return {"twin_vs_coeff_flip_max_abs": coeff_err, "twin_intensity_max_rel": intensity_rel_err}

def evaluate(
    checkpoint: str,
    num_samples: int = 1000,
    seed: int = 1234,
    mode: str = 'hybrid',
    batch_size: int = 64,
    out_dir: str = 'eval'
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = OpticsConfig()
    model = UNet(cfg, in_channels=1, out_channels=1, mode=mode).to(device)
    load_checkpoint_strict(model, checkpoint, device)
    model.eval()
    print(f"Loaded {checkpoint} (mode={mode}) on {device}")

    random.seed(seed)
    torch.manual_seed(seed)
    dataset = PhaseRetrievalDataset(cfg, num_samples=num_samples, device=torch.device('cpu'), return_coeffs=True)
    noll = np.array(dataset.noll_indices)
    is_even = np.array([abs(noll_to_nm(int(j))[1]) % 2 == 0 for j in noll])
    checks = self_check(dataset, is_even)
    print(f"Self-check passed: {checks}")

    # Least-squares projection of a pupil phase onto [piston, basis]. Truth is piston-removed
    # after synthesis, so including piston recovers the synthesis coefficients exactly.
    pupil = dataset.mask > 0.5
    piston = get_noll_polynomial(1, dataset.rho, dataset.theta, dataset.mask)
    design = torch.cat([piston[pupil].unsqueeze(1), dataset.basis[:, pupil].T], dim=1).double()
    projector = torch.linalg.pinv(design)  # [1 + M, P]

    def project(phases: torch.Tensor) -> torch.Tensor:
        return (projector @ phases[:, pupil].T.double()).T[:, 1:].float()

    true_c, head_c, proj_c, truth_proj_c = [], [], [], []
    rmse_truth, rmse_twin, rms_true = [], [], []
    rms_even_true, rms_even_pred, rmse_odd = [], [], []

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            batch = [dataset[i] for i in range(start, min(start + batch_size, num_samples))]
            intensities = torch.stack([b[0] for b in batch]).to(device)
            truths = torch.stack([b[1] for b in batch]).squeeze(1)  # [B, N, N]
            coeffs = torch.stack([b[2] for b in batch])

            pred, pred_coeffs = model(intensities, return_coeffs=True)
            pred = pred.squeeze(1).cpu() * dataset.mask

            twins = twin_phase(truths) * dataset.mask
            err_truth = (pred - truths)[:, pupil]
            err_twin = (pred - twins)[:, pupil]
            rmse_truth.append(err_truth.pow(2).mean(1).sqrt())
            rmse_twin.append(err_twin.pow(2).mean(1).sqrt())
            rms_true.append(truths[:, pupil].pow(2).mean(1).sqrt())

            # Pixelwise even/odd split: phi_e = (phi + phi(-r))/2 flips sign under the twin map, phi_o does not
            def split(p):
                p_neg = -twin_phase(p)  # phi(-r)
                return 0.5 * (p + p_neg), 0.5 * (p - p_neg)
            true_e, true_o = split(truths)
            pred_e, pred_o = split(pred)
            rms_even_true.append(true_e[:, pupil].pow(2).mean(1).sqrt())
            rms_even_pred.append(pred_e[:, pupil].pow(2).mean(1).sqrt())
            rmse_odd.append((pred_o - true_o)[:, pupil].pow(2).mean(1).sqrt())

            true_c.append(coeffs)
            head_c.append(pred_coeffs.cpu())
            proj_c.append(project(pred))
            truth_proj_c.append(project(truths))

    cat = lambda xs: torch.cat(xs).numpy()
    true_c, head_c, proj_c, truth_proj_c = map(cat, (true_c, head_c, proj_c, truth_proj_c))
    rmse_truth, rmse_twin, rms_true = map(cat, (rmse_truth, rmse_twin, rms_true))
    rms_even_true, rms_even_pred, rmse_odd = map(cat, (rms_even_true, rms_even_pred, rmse_odd))

    proj_sanity = np.abs(truth_proj_c - true_c).max()
    assert proj_sanity < 1e-3, f"Projection does not recover true coefficients (max err {proj_sanity:.2e})"

    # Twin-aware scoring: per sample, score against whichever of (c, c_twin) the prediction is closer to.
    # The twin flips all even coefficients jointly, so this measures magnitudes and relative even signs.
    sign = np.where(is_even, -1.0, 1.0)
    twin_c = true_c * sign
    use_twin = ((proj_c - twin_c) ** 2).sum(1) < ((proj_c - true_c) ** 2).sum(1)
    aligned_c = np.where(use_twin[:, None], twin_c, true_c)

    r2_head = r2_score(head_c, true_c)
    r2_proj = r2_score(proj_c, true_c)
    r2_twin_aware = r2_score(proj_c, aligned_c)
    rmse_proj = np.sqrt(((proj_c - true_c) ** 2).mean(0))

    twin_closer = rmse_twin < rmse_truth
    margin = np.abs(rmse_twin - rmse_truth) / np.maximum(np.minimum(rmse_twin, rmse_truth), 1e-8)

    print("\nPer-mode coefficient accuracy (projected = LSQ fit of total predicted phase)")
    print(f"{'Noll':>4} {'(n,m)':>7} {'parity':>6} {'R2 head':>8} {'R2 proj':>8} {'R2 twin':>8} {'RMSE':>7} {'std true':>8}")
    for k, j in enumerate(noll):
        n, m = noll_to_nm(int(j))
        print(f"{j:>4} {f'({n},{m})':>7} {'even' if is_even[k] else 'odd':>6} "
              f"{r2_head[k]:>8.3f} {r2_proj[k]:>8.3f} {r2_twin_aware[k]:>8.3f} "
              f"{rmse_proj[k]:>7.3f} {true_c[:, k].std():>8.3f}")
    print(f"\nMean R2 (projected): odd {r2_proj[~is_even].mean():.3f} | even {r2_proj[is_even].mean():.3f}")
    print(f"Mean R2 (twin-aware): odd {r2_twin_aware[~is_even].mean():.3f} | even {r2_twin_aware[is_even].mean():.3f}")

    print("\nPhase RMSE inside pupil [rad] (median / mean)")
    print(f"  vs truth:            {np.median(rmse_truth):.3f} / {rmse_truth.mean():.3f}")
    print(f"  vs twin -phi(-r):    {np.median(rmse_twin):.3f} / {rmse_twin.mean():.3f}")
    print(f"  flat-prediction ref: {np.median(rms_true):.3f} / {rms_true.mean():.3f}  (RMS of true phase)")
    print(f"  min(truth, twin):    {np.median(np.minimum(rmse_truth, rmse_twin)):.3f}")
    print(f"Twin closer than truth: {twin_closer.mean():.1%} of samples "
          f"({(twin_closer & (margin > 0.1)).mean():.1%} by >10% margin; "
          f"truth closer by >10%: {(~twin_closer & (margin > 0.1)).mean():.1%})")
    print("\nEven/odd pixelwise split (median)")
    print(f"  RMS even part: true {np.median(rms_even_true):.3f} | predicted {np.median(rms_even_pred):.3f} "
          f"(ratio {np.median(rms_even_pred / rms_even_true):.3f})")
    print(f"  RMSE of odd part: {np.median(rmse_odd):.3f}")

    # Per-mode R2 bar chart, even vs odd colored
    surface, ink, muted = '#fcfcfb', '#0b0b0b', '#52514e'
    odd_color, even_color = '#2a78d6', '#eb6834'
    x = np.arange(len(noll))
    floor = -0.5
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=surface)
    ax.set_facecolor(surface)
    colors = [even_color if e else odd_color for e in is_even]
    ax.bar(x - 0.2, np.clip(r2_proj, floor, 1), width=0.38, color=colors, edgecolor=surface, linewidth=1)
    ax.bar(x + 0.2, np.clip(r2_twin_aware, floor, 1), width=0.38, color=colors, alpha=0.45,
           hatch='//', edgecolor=surface, linewidth=1)
    for k in np.where(r2_proj < floor)[0]:
        ax.annotate(f"{r2_proj[k]:.1f}", (x[k] - 0.2, floor), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=7, color=muted)
    ax.axhline(0, color=muted, linewidth=0.8)
    ax.set_xticks(x, [str(j) for j in noll])
    ax.set_xlabel('Noll index', color=muted)
    ax.set_ylabel('R²', color=muted)
    ax.set_ylim(floor, 1.05)
    ax.grid(axis='y', color='#e4e3df', linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
    ax.set_title(f"Per-mode coefficient R² — {os.path.basename(checkpoint)}, {num_samples} samples",
                 color=ink, loc='left')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=odd_color, label='odd |m| (twin-invariant)'),
        Patch(color=even_color, label='even |m| (sign flips under twin)'),
        Patch(facecolor='#bbbbbb', hatch='//', edgecolor=surface, label='hatched: twin-aware R²'),
    ], frameon=False, loc='lower left', fontsize=8, labelcolor=muted)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'per_mode_r2.png')
    plt.savefig(fig_path, dpi=150, facecolor=surface)
    plt.close(fig)

    metrics = {
        "checkpoint": checkpoint, "mode": mode, "num_samples": num_samples, "seed": seed,
        "self_check": checks,
        "noll": noll.tolist(), "is_even": is_even.tolist(),
        "r2_head": r2_head.tolist(), "r2_proj": r2_proj.tolist(), "r2_twin_aware": r2_twin_aware.tolist(),
        "rmse_proj": rmse_proj.tolist(),
        "phase_rmse_truth_median": float(np.median(rmse_truth)),
        "phase_rmse_twin_median": float(np.median(rmse_twin)),
        "true_phase_rms_median": float(np.median(rms_true)),
        "twin_closer_fraction": float(twin_closer.mean()),
        "even_rms_ratio_median": float(np.median(rms_even_pred / rms_even_true)),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {fig_path} and {os.path.join(out_dir, 'metrics.json')}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-mode and twin-image evaluation of a phase retrieval checkpoint")
    parser.add_argument('--checkpoint', type=str, default='saved_models/unet_phase_retrieval_best.pth')
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mode', type=str, default='hybrid', choices=['hybrid', 'modal', 'unet'],
                        help="Architecture mode the checkpoint was trained with (not stored in legacy checkpoints)")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out-dir', type=str, default='eval')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.num_samples, args.seed, args.mode, args.batch_size, args.out_dir)
