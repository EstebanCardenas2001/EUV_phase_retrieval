"""
Regenerates the README figures in docs/figures/ from the evaluation outputs in eval/ (and, for the
two illustrative figures, from the simulator and the final checkpoints). Run from the repo root:

    python docs/make_figures.py
"""
import json
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.config import OpticsConfig
from physics.simulator import OpticalSystem, twin_phase
from physics.zernike import zernike_polynomial
from ai.evaluate import rad_to_nm, even_mode_mask, marechal_strehl
from ai.dataset import PhaseRetrievalDataset

OUT = 'docs/figures'
FINAL = [f'saved_models/final/m{s}/best.pth' for s in range(5)]

# Palette (validated categorical slots + neutral text/surface tokens)
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
GRAY = '#a8a7a2'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'normal',
    'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
})

def load(path):
    with open(path) as f:
        return json.load(f)

def style(ax, grid_axis='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

def clean_image_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")

def center(img, w=40):
    c = img.shape[-1] // 2
    return img[..., c - w:c + w, c - w:c + w]

# 1. The twin-image problem ------------------------------------------------------------------------
def twin_problem():
    cfg = OpticsConfig()
    sim = OpticalSystem(cfg)
    geo = cfg.build_geometry()
    phi = (zernike_polynomial(geo.rho, geo.theta, geo.mask, 'astigmatism_vertical') * 0.816 +
           zernike_polynomial(geo.rho, geo.theta, geo.mask, 'coma_horizontal') * -0.530 +
           zernike_polynomial(geo.rho, geo.theta, geo.mask, 'defocus') * 0.45)
    phis = torch.stack([phi, twin_phase(phi) * geo.mask])
    with torch.no_grad():
        stack = sim(phis)  # [2, 3, N, N]
    rel = [((stack[0, k] - stack[1, k]).norm() / stack[0, k].norm()).item() for k in range(3)]
    shown = torch.log10(center(stack) + 1e-2).numpy()
    vmin, vmax = shown.min(), shown.max()
    pupil = geo.mask.numpy() > 0.5
    lim = float(np.abs(phis.numpy()[:, pupil]).max())

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), gridspec_kw={'width_ratios': [1.15, 1, 1, 1]})
    rows = ["Wavefront φ", "Its twin −φ(−r)"]
    cols = ["defocus −1 rad", "in focus", "defocus +1 rad"]
    for r in range(2):
        ph = np.where(pupil, phis[r].numpy(), np.nan)
        im = axes[r, 0].imshow(center(ph, 64), cmap='RdBu_r', vmin=-lim, vmax=lim)
        clean_image_axis(axes[r, 0])
        axes[r, 0].set_ylabel(rows[r], color=INK, fontsize=12, labelpad=8)
        for k in range(3):
            axes[r, k + 1].imshow(shown[r, k], cmap='magma', vmin=vmin, vmax=vmax)
            clean_image_axis(axes[r, k + 1])
    axes[0, 0].set_title("Pupil phase (hidden)", color=INK)
    for k in range(3):
        axes[0, k + 1].set_title(f"Detector image, {cols[k]}", color=INK)
        verdict = "identical" if rel[k] < 1e-4 else "different"
        color = ORANGE if rel[k] < 1e-4 else AQUA
        axes[1, k + 1].text(0.5, -0.1, verdict, transform=axes[1, k + 1].transAxes, ha='center', va='top',
                            color=color, fontsize=12, fontweight='bold')
    cb = fig.colorbar(im, ax=axes[1, 0], fraction=0.06, pad=0.06, location='bottom')
    cb.set_label('phase [rad]', color=MUTED)
    fig.suptitle("One in-focus image cannot tell a wavefront from its twin. Defocused images can.",
                 color=INK, fontsize=13, x=0.52, y=1.0)
    save(fig, 'twin_problem.png')

# 2. Per-mode R2: original single-image model vs final ensemble ---------------------------------------
def per_mode_before_after():
    before = load('eval/metrics.json')          # original single in-focus image model (Phase 0)
    after = load('eval/final_ens5/metrics.json')
    noll = np.array(after['noll'])
    even = even_mode_mask(noll)
    x = np.arange(len(noll))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i in np.where(even)[0]:
        ax.axvspan(i - 0.5, i + 0.5, color='#fbe7dc', zorder=0, linewidth=0)
    ax.bar(x - 0.2, np.array(before['r2_proj']), width=0.38, color=GRAY, label='1 in-focus image (original model)')
    ax.bar(x + 0.2, np.array(after['r2_proj']), width=0.38, color=BLUE, label='3 diversity images (final ensemble)')
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [str(j) for j in noll])
    ax.set_xlim(-0.6, len(noll) - 0.4)
    ax.set_ylim(-0.15, 1.08)
    ax.set_xlabel('Zernike mode (Noll index)')
    ax.set_ylabel('R² of recovered coefficient')
    ax.text(-0.5, 1.03, 'shaded = even modes (defocus, astigmatism, spherical, ...): their sign is lost in a single image',
            color=ORANGE, fontsize=9, transform=ax.get_xaxis_transform())
    style(ax)
    ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.06), ncol=2, labelcolor=MUTED, fontsize=9)
    ax.set_title('Every mode is recovered once the twin ambiguity is broken', color=INK, loc='left', pad=34)
    save(fig, 'per_mode_before_after.png')

# 3. Example reconstruction with the final ensemble ---------------------------------------------------
def reconstruction_example(seed=7):
    from ai.checkpoint import load_models
    from ai.ensemble import Ensemble
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, cfg, _ = load_models(FINAL, device)
    model = Ensemble(models).eval()
    random.seed(seed)
    torch.manual_seed(seed)
    ds = PhaseRetrievalDataset(cfg, num_samples=1, device=device)
    ds.noise_aug_max = 1.0
    x, truth, _ = ds[0]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze().cpu()
    truth = truth.squeeze().cpu()
    pupil = (ds.mask.cpu() > 0.5)
    err = (pred - truth)
    err = err - err[pupil].mean()
    rms = err[pupil].pow(2).mean().sqrt().item()
    lim = float(truth[pupil].abs().max())
    to_img = lambda t: center(np.where(pupil.numpy(), t.numpy(), np.nan), 64)

    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.1, 1, 1, 1], wspace=0.18)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(np.concatenate(list(x.cpu().numpy()), axis=1), cmap='magma')
    clean_image_axis(ax0)
    ax0.set_title('Input: 3 detector images (defocus −1, 0, +1)', color=INK)
    axes = [fig.add_subplot(gs[i]) for i in (1, 2, 3)]
    axes[0].imshow(to_img(truth), cmap='RdBu_r', vmin=-lim, vmax=lim)
    axes[0].set_title('True wavefront', color=INK)
    im = axes[1].imshow(to_img(pred), cmap='RdBu_r', vmin=-lim, vmax=lim)
    axes[1].set_title('Recovered (network)', color=INK)
    err_nm = rad_to_nm(err)
    elim = float(np.nanmax(np.abs(to_img(err_nm))))
    im2 = axes[2].imshow(to_img(err_nm), cmap='RdBu_r', vmin=-elim, vmax=elim)
    axes[2].set_title(f'Error: {rad_to_nm(rms):.2f} nm RMS', color=INK)
    for a in axes:
        clean_image_axis(a)
    c1 = fig.colorbar(im, ax=axes[:2], fraction=0.025, pad=0.03)
    c1.set_label('phase [rad]', color=MUTED)
    c2 = fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.04)
    c2.set_label('error [nm]', color=MUTED)
    save(fig, 'reconstruction_example.png')
    return rms

# 4. Model progression -------------------------------------------------------------------------------
def model_progression():
    models = [
        ('Original\n1 image', 'eval/metrics.json', None),
        ('3 images\n(k3_pm1)', 'eval/k3_pm1/metrics.json', 'eval/k3_pm1_n10/metrics.json'),
        ('+ noise aug.\n5-model ensemble', 'eval/ens5/metrics.json', 'eval/ens5_n10/metrics.json'),
        ('Final\nsingle model', 'eval/final_m0/metrics.json', 'eval/final_m0_n10/metrics.json'),
        ('Final\n5-model ensemble', 'eval/final_ens5/metrics.json', 'eval/final_ens5_n10/metrics.json'),
    ]
    x = np.arange(len(models))
    nominal = [rad_to_nm(load(a)['phase_rmse_truth_median']) for _, a, _ in models]
    noisy = [rad_to_nm(load(b)['phase_rmse_truth_median']) if b else np.nan for _, _, b in models]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    b1 = ax.bar(x - 0.2, nominal, width=0.38, color=BLUE, label='nominal noise')
    b2 = ax.bar(x + 0.2, noisy, width=0.38, color=ORANGE, label='10× noise')
    for bars, vals in ((b1, nominal), (b2, noisy)):
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.08, f'{v:.2f}', ha='center', va='bottom',
                        fontsize=8.5, color=INK)
    ax.text(x[0] + 0.2, 0.06, 'not\ntested', ha='center', fontsize=8, color=MUTED)
    ax.set_yscale('log')
    ax.set_ylim(0.05, 6)
    ax.set_xticks(x, [m[0] for m in models])
    ax.set_ylabel('median wavefront error [nm RMS]')
    style(ax)
    ax.legend(frameon=False, labelcolor=MUTED, loc='upper right')
    ax.set_title('Wavefront error at 13.5 nm, 1000 test samples (lower is better, log scale)', color=INK, loc='left')
    save(fig, 'model_progression.png')
    return nominal, noisy

# 5. Speed vs accuracy -------------------------------------------------------------------------------
def speed_accuracy():
    ens = load('eval/benchmark/benchmark.json')['results']
    one = load('eval/benchmark_final_m0/benchmark.json')['results']
    pts = [
        ('Network (1 model)', one['network (batched)'], BLUE, (10, 6)),
        ('Network (5-model ensemble)', ens['network (batched)'], BLUE, (10, 6)),
        ('Physics solver, 300 iterations', one['solver 300 it (batched)'], ORANGE, (-30, 14)),
        ('Network → solver, 50 iterations', one['network -> solver 50 it'], AQUA, (10, -30)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for label, r, color, offset in pts:
        ax.scatter(r['ms_per_sample'], r['median_wrapped_rmse_nm'], s=110, color=color, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(f"{label}\n{r['median_wrapped_rmse_nm']:.3f} nm, {r['ms_per_sample']:.0f} ms",
                    (r['ms_per_sample'], r['median_wrapped_rmse_nm']), xytext=offset,
                    textcoords='offset points', fontsize=8.5, color=INK)
    ax.set_xscale('log')
    ax.set_xlim(3, 900)
    ax.set_ylim(0, 0.19)
    ax.set_xlabel('time per sample [ms], Tesla T4, batched (log scale)')
    ax.set_ylabel('median wavefront error [nm RMS]')
    style(ax, grid_axis='both')
    ax.set_title('Speed vs accuracy: the network gives the solver a head start', color=INK, loc='left')
    save(fig, 'speed_accuracy.png')

# 6. Uncertainty coverage ----------------------------------------------------------------------------
def uncertainty_coverage():
    u = load('eval/final_ens5/uq_calibration.json')['sets']
    sets = [('test', 'test'), ('noise x10', '10× noise'), ('noise x100', '100× noise\n(beyond training)'),
            ('OOD coeffs x1.5', 'aberrations 1.5×\nlarger than training')]
    x = np.arange(len(sets))
    c1 = [100 * u[k]['pixel']['raw']['coverage_1sigma'] for k, _ in sets]
    c2 = [100 * u[k]['pixel']['raw']['coverage_2sigma'] for k, _ in sets]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, c1, width=0.38, color=BLUE, label='inside ±1σ')
    ax.bar(x + 0.2, c2, width=0.38, color=AQUA, label='inside ±2σ')
    for xi, a, b in zip(x, c1, c2):
        ax.text(xi - 0.2, a + 1.5, f'{a:.0f}%', ha='center', fontsize=8.5, color=INK)
        ax.text(xi + 0.2, b + 1.5, f'{b:.0f}%', ha='center', fontsize=8.5, color=INK)
    ax.axhline(68.3, color=BLUE, linestyle='--', linewidth=1)
    ax.axhline(95.4, color=AQUA, linestyle='--', linewidth=1)
    ax.set_xlim(-0.6, len(sets) - 0.4 + 0.55)
    ax.text(len(sets) - 0.35, 68.3, 'ideal\n68%', color=BLUE, fontsize=8, ha='left', va='center')
    ax.text(len(sets) - 0.35, 95.4, 'ideal\n95%', color=AQUA, fontsize=8, ha='left', va='center')
    ax.set_xticks(x, [s[1] for s in sets])
    ax.set_ylim(0, 108)
    ax.set_ylabel('true phase inside predicted interval [%]')
    style(ax)
    ax.legend(frameon=False, labelcolor=MUTED, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=9)
    ax.set_title('Uncertainty (5-model ensemble): calibrated, except for aberrations larger than trained on',
                 color=INK, loc='left', pad=26)
    save(fig, 'uncertainty_coverage.png')

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    twin_problem()
    per_mode_before_after()
    rms = reconstruction_example()
    print(f"reconstruction example: residual {rms:.4f} rad = {rad_to_nm(rms):.4f} nm")
    nominal, noisy = model_progression()
    print("model progression [nm]:", [round(v, 3) for v in nominal], [round(v, 3) if np.isfinite(v) else None for v in noisy])
    speed_accuracy()
    uncertainty_coverage()
