#!/bin/bash
#SBATCH --job-name=robot-imitation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/q3-%A_%a.out
#SBATCH --error=logs/q3-%A_%a.err

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

python -c 'import torch; assert torch.cuda.is_available(), "No usable CUDA/ROCm GPU on allocated node"; print("GPU:",torch.cuda.get_device_name(0),"CUDA:",torch.version.cuda,"ROCm:",torch.version.hip)'

common=(--device cuda --seed 0
        --batch_size "${BATCH_SIZE:-64}"
        --gradient_steps "${GRADIENT_STEPS:-100000}"
        --eval_episodes "${EVAL_EPISODES:-25}"
        --max_episode_steps "${MAX_EPISODE_STEPS:-500}")
results_root="${RESULTS_ROOT:-results}"

case "$SLURM_ARRAY_TASK_ID" in
  0) srun python -u bc.py "${common[@]}" --results_dir "$results_root/bc" ;;
  1) srun python -u diffusion_policy.py "${common[@]}" --results_dir "$results_root/diffusion_policy_eps" ;;
  2) srun python -u diffusion_policy.py --predict_x0 "${common[@]}" --results_dir "$results_root/diffusion_policy_x0" ;;
esac
