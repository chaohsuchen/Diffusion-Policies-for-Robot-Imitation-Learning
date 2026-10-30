# Visual robot imitation: BC and diffusion policies

The main project overview is in the [root README](../README.md). See the [walkthrough](../docs/PROJECT_REPORT.md) for architecture and findings and the [reproduction guide](../docs/REPRODUCING.md) for setup, dataset download, training, and evaluation.

This directory contains the course-derived Robomimic square-task pipeline: `bc.py` predicts one action, while `diffusion_policy.py` generates action chunks with `ddpm.py`. Both use the same observation encoder and backbone architecture. The default dataset is `data/robomimic/datasets/square/ph/image.hdf5`, relative to this directory.

Run from this directory after setup:

```bash
python bc.py
python diffusion_policy.py
python diffusion_policy.py --predict_x0
python diffusion_policy.py --beta_schedule sigmoid
```

Formal evaluation CSVs and sampling videos are in `results/`. The default budget is 100,000 updates; avoid reusing a results directory for an independent run because the logger appends metrics and overwrites weights. Dataset files and neural checkpoints are excluded from version control.
