Benchmark: ensemble of 5, 256 samples, noise x1, Tesla T4

| Method | Median RMSE [rad] | [nm] | ms / sample | Note |
|---|---|---|---|---|
| network (batched) | 0.0464 | 0.0998 | 27.75 | batch 64 |
| network (per sample) | 0.0464 | 0.0998 | 33.38 | batch 1, median of 8 |
| solver 300 it (batched) | 0.0317 | 0.0681 | 161.88 | all 256 samples in one batch |
| network -> solver 50 it | 0.0297 | 0.0638 | 54.97 | network batched + batched refinement |
