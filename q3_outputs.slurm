#!/bin/bash
#SBATCH --job-name=robot-imitation-outputs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/q3-outputs-%j.out
#SBATCH --error=logs/q3-outputs-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export MUJOCO_GL="${RENDER_BACKEND:-egl}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$SLURM_SUBMIT_DIR/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SLURM_SUBMIT_DIR/.cache/matplotlib}"
cd Q3

for name in bc diffusion_policy_eps diffusion_policy_x0; do
  test -s "results/$name/eval_metrics.csv"
done
python plot_completion_vs_gradient_steps.py
python plot_diffusion_generation.py --device cuda
python plot_diffusion_generation.py --predict_x0 --device cuda
