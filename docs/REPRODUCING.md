# Reproduce the robot imitation experiments

Run commands from the repository root unless otherwise specified. Full training requires a downloaded image dataset and substantial compute. The included evaluation CSVs let you inspect the findings without retraining. Checkpoints are not distributed.

## Environment

Use Python 3.10. Create a local environment and install dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps \
  'robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0'
```

For a Linux training node, install a torch/torchvision pair compatible with its GPU first, then use `requirements-q3-server.txt` instead of `requirements.txt` and install the same robomimic revision. The server requirements preserve the simulator-stack versions used by the project; the general requirements are less strictly pinned. This repository does not contain a complete original environment lockfile.

On Linux, rendering defaults to EGL. Use `MUJOCO_GL=osmesa` only with a working OSMesa installation. On macOS the code selects the native rendering backend. Device selection uses CUDA when available, then Apple MPS, then CPU.

## Download the demonstrations

```bash
mkdir -p Q3/data/robomimic/datasets/square/ph
curl -fL -o Q3/data/robomimic/datasets/square/ph/image.hdf5 \
  https://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/image.hdf5
```

This is the square-task proficient-human image dataset (about 2.4 GiB locally). The inspected copy contains 200 demonstrations and 30,154 action steps. The loader uses all demonstrations, not the built-in train/validation masks.

## Check execution before a long run

```bash
python smoke_q3.py --device mps   # or --device cuda / --device cpu
```

The smoke check creates a separate two-demo subset, runs three updates per model, checks checkpoint reload/inference, and generates diagnostic outputs. It checks execution, not policy quality. Its output is kept outside the formal results folders.

## Train the four configurations

From `Q3/`, using fresh output directories for independent runs:

```bash
cd Q3
python bc.py
python diffusion_policy.py
python diffusion_policy.py --predict_x0
python diffusion_policy.py --beta_schedule sigmoid
```

Standard settings: seed 0, 100,000 updates, batch size 64, learning rate 1e-4, evaluation every 10,000 updates on 25 episodes, maximum 500 environment steps. Diffusion uses 50 denoising steps, prediction horizon 10, and execution interval 8. BC uses a single action. These are code/script defaults; archived CSVs do not store a complete command-line manifest.

The logger appends CSV records and overwrites matching checkpoint filenames. Move archived results aside or supply `--results_dir` before rerunning. For GPU scheduling, the root `q3.slurm` runs the three main configurations as an array; `q3_outputs.slurm` generates their plot and sampling videos. Supply your cluster's partition/account/resource choices at submission. `Q3/train_4090.slurm` is an optional single-model NVIDIA job template.

## Inspect and regenerate outputs

```bash
# From repository root, with matplotlib installed:
python scripts/analyze_results.py

# From Q3, after training and with matching checkpoints:
python plot_completion_vs_gradient_steps.py
python plot_diffusion_generation.py
python plot_diffusion_generation.py --predict_x0
```

`docs/assets/learning_curves.png` keeps the main comparison separate from the bonus. `docs/results_summary.json` records final and best-checkpoint values. The original plotting script detects every run under `Q3/results/` and includes them in one figure.

The sampling videos plot action vectors during reverse diffusion; they do not record task success. Success-rate comparisons should use `eval_metrics.csv`, not raw accumulated rewards, whose aggregation differs between BC and diffusion.

## Rebuild the portfolio PDF

```bash
python -m pip install -r docs/requirements-report.txt
python scripts/build_report.py
```

The four-page report is written to `output/pdf/robot_imitation_report.pdf`. It uses the same archived evaluation CSVs as the analysis. Its diagrams and curve are vector graphics generated with ReportLab.

## Separate DMP exercise

`Q2/` retains the course-derived UR10 ball-catching work. Install its environment with `python -m pip install -e Q2/ballcatch_env/`, then run its scripts from that directory. It is separate from the image-based square-nut experiments and is not used to support the reported diffusion results.
