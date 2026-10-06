import torch
import numpy as np
import argparse
import json
import random
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.checkpoint import load_models
from ai.ensemble import Ensemble
from ai.dataset import PhaseRetrievalDataset, preprocess_intensity
from ai.evaluate import rad_to_nm
from solver.gradient_descent import solve_phase, wrapped_pupil_rmse

def make_samples(cfg, num_samples, seed, noise_mult, device):
    """Fixed-noise test samples: true phases, raw K-plane intensities (solver) and preprocessed inputs (network)."""
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = PhaseRetrievalDataset(cfg, num_samples=1, device=device)
    phases = torch.stack([dataset.sample_phase()[0] for _ in range(num_samples)])
    with torch.no_grad():
        clean = dataset.simulator(phases)
        noise = torch.randn_like(clean) * dataset.noise_std * noise_mult
        intensity = torch.clamp(clean + noise, min=0.0)
        inputs = preprocess_intensity(intensity, cfg.crop_size)
    return dataset.simulator, dataset.mask, phases, intensity, inputs

def timed(fn, device):
    """Wall-clock seconds of fn(), synchronizing the GPU before and after so queued kernels are counted."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0

def run(checkpoints, num_samples=64, seed=1234, noise_mult=1.0, solver_iters=300, refine_iters=50,
        batch_size=64, latency_samples=8, out_dir='eval/benchmark'):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, cfg, ckpts = load_models(checkpoints, device)
    model = models[0] if len(models) == 1 else Ensemble(models).eval()
    label = os.path.relpath(checkpoints[0]) if len(checkpoints) == 1 else f"ensemble of {len(checkpoints)}"
    sim, mask, phases, intensity, inputs = make_samples(cfg, num_samples, seed, noise_mult, device)
    print(f"{label} | {num_samples} samples (seed {seed}, noise x{noise_mult:g}) | K={cfg.K} | "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    def network(x):
        with torch.no_grad():
            return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]).squeeze(1)

    # Warm-up (CUDA context, cuDNN algorithm selection, FFT plans) so first-call overhead is not timed
    network(inputs[:batch_size])
    solve_phase(sim, intensity[:2], iterations=3, verbose=False)

    results = {}

    def record(name, pred, seconds, n_timed, note):
        err = wrapped_pupil_rmse(pred, phases[:len(pred)], mask).cpu().numpy()
        results[name] = {
            "median_wrapped_rmse_rad": float(np.median(err)), "mean_wrapped_rmse_rad": float(err.mean()),
            "median_wrapped_rmse_nm": float(rad_to_nm(np.median(err))),
            "seconds_total": float(seconds), "ms_per_sample": float(1e3 * seconds / n_timed), "note": note,
        }

    # 1. Network, batched over all samples
    pred_nn, t = timed(lambda: network(inputs), device)
    record("network (batched)", pred_nn, t, num_samples, f"batch {batch_size}")

    # 2. Network, one sample at a time (latency); median over samples
    times = []
    for i in range(latency_samples):
        _, t = timed(lambda: network(inputs[i:i + 1]), device)
        times.append(t)
    results["network (per sample)"] = {**results["network (batched)"],
                                       "seconds_total": float(np.sum(times)),
                                       "ms_per_sample": float(1e3 * np.median(times)),
                                       "note": f"batch 1, median of {latency_samples}"}

    # 3. Pixelwise solver from zero, batched (amortized), and single-sample latency
    torch.manual_seed(0)
    pred_solver, t = timed(lambda: solve_phase(sim, intensity, iterations=solver_iters, verbose=False)[0], device)
    record(f"solver {solver_iters} it (batched)", pred_solver, t, num_samples, f"all {num_samples} samples in one batch")
    _, t1 = timed(lambda: solve_phase(sim, intensity[:1], iterations=solver_iters, verbose=False), device)
    results[f"solver {solver_iters} it (batched)"]["single_sample_seconds"] = float(t1)

    # 4. Hybrid: network prediction as the solver's starting point, short refinement
    def hybrid():
        start = network(inputs)
        return solve_phase(sim, intensity, iterations=refine_iters, verbose=False, init_phase=start)[0]
    torch.manual_seed(0)
    pred_hybrid, t = timed(hybrid, device)
    record(f"network -> solver {refine_iters} it", pred_hybrid, t, num_samples, "network batched + batched refinement")

    print(f"\n{'method':>26} | {'median RMSE':>11} {'[nm]':>7} | {'ms/sample':>9} | note")
    for name, r in results.items():
        print(f"{name:>26} | {r['median_wrapped_rmse_rad']:>11.4f} {r['median_wrapped_rmse_nm']:>7.4f} | "
              f"{r['ms_per_sample']:>9.2f} | {r['note']}")
    print(f"(solver single-sample latency, {solver_iters} it: {results[f'solver {solver_iters} it (batched)']['single_sample_seconds']:.2f} s;"
          f" RMSE is wrapped mod 2pi and piston-free)")

    summary = {
        "checkpoint": list(checkpoints), "num_samples": num_samples, "seed": seed, "noise_mult": noise_mult,
        "solver_iters": solver_iters, "refine_iters": refine_iters, "batch_size": batch_size,
        "device": torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', "results": results,
    }
    with open(os.path.join(out_dir, 'benchmark.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, 'benchmark.md'), 'w') as f:
        f.write(f"Benchmark: {label}, {num_samples} samples, noise x{noise_mult:g}, {summary['device']}\n\n")
        f.write("| Method | Median RMSE [rad] | [nm] | ms / sample | Note |\n|---|---|---|---|---|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['median_wrapped_rmse_rad']:.4f} | {r['median_wrapped_rmse_nm']:.4f} | "
                    f"{r['ms_per_sample']:.2f} | {r['note']} |\n")
    print(f"Saved {out_dir}/benchmark.json and benchmark.md")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speed / accuracy: network vs pixelwise solver vs network -> solver")
    parser.add_argument('--checkpoint', type=str, nargs='+', default=['saved_models/latest/best.pth'],
                        help="One checkpoint, or several ensemble members (same config)")
    parser.add_argument('--num-samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--noise-mult', type=float, default=1.0)
    parser.add_argument('--solver-iters', type=int, default=300)
    parser.add_argument('--refine-iters', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out-dir', type=str, default='eval/benchmark')
    args = parser.parse_args()
    run(args.checkpoint, args.num_samples, args.seed, args.noise_mult, args.solver_iters, args.refine_iters,
        args.batch_size, out_dir=args.out_dir)
