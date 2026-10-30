#!/bin/bash
#SBATCH --job-name=q3-4090
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
# Submit from Q3/ and supply your cluster account/partition to sbatch.
cd "$SLURM_SUBMIT_DIR"
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

MODEL="${1:-diffusion_policy}"
case "$MODEL" in
    diffusion_policy|bc) ;;
    *) echo 'Usage: sbatch train_4090.slurm [diffusion_policy|bc] [training arguments...]' >&2; exit 2 ;;
esac
if (( $# > 0 )); then shift; fi

python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
assert torch.version.cuda is not None, 'Install a CUDA build of PyTorch first.'
assert torch.cuda.is_available(), 'NVIDIA GPU is unavailable; check the driver and Slurm allocation.'
x = torch.ones(4, device='cuda')
print('GPU:', torch.cuda.get_device_name(0), 'test:', (x + x).cpu(), flush=True)
import torchvision, mujoco, robosuite, robomimic, h5py
with h5py.File('data/robomimic/datasets/square/ph/image.hdf5', 'r') as f:
    assert 'data' in f, 'Dataset is missing its data group.'
    print('Dataset demonstrations:', len(f['data']), flush=True)
PY

srun python -u "${MODEL}.py" --device cuda \
    --results_dir "results/${MODEL}-${SLURM_JOB_ID}" "$@"
