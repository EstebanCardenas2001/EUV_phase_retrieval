Benchmark: saved_models/final/m0/best.pth, 256 samples, noise x1, Tesla T4

| Method | Median RMSE [rad] | [nm] | ms / sample | Note |
|---|---|---|---|---|
| network (batched) | 0.0740 | 0.1591 | 5.58 | batch 64 |
| network (per sample) | 0.0740 | 0.1591 | 6.50 | batch 1, median of 8 |
| solver 300 it (batched) | 0.0317 | 0.0681 | 161.95 | all 256 samples in one batch |
| network -> solver 50 it | 0.0296 | 0.0636 | 32.65 | network batched + batched refinement |
