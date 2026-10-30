"""Check Q3 end to end using two real demonstrations and three updates per model."""

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    q3 = root / "Q3"
    run_dir = q3 / "smoke_results" / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    logs = run_dir / "logs"
    logs.mkdir()
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("NUMBA_CACHE_DIR", str(run_dir / "cache" / "numba"))
    env.setdefault("MPLCONFIGDIR", str(run_dir / "cache" / "matplotlib"))

    import h5py
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; use --device cpu for the local smoke check")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU unavailable in this process")

    source = q3 / "data/robomimic/datasets/square/ph/image.hdf5"
    subset = run_dir / "image_subset.hdf5"
    if not source.is_file():
        raise FileNotFoundError(f"Download the square image dataset first: {source}")
    with h5py.File(source, "r") as src, h5py.File(subset, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        data = dst.create_group("data")
        for key, value in src["data"].attrs.items():
            data.attrs[key] = value
        total = 0
        for i in range(2):
            src.copy(src[f"data/demo_{i}"], data, name=f"demo_{i}")
            total += len(data[f"demo_{i}/actions"])
        data.attrs["total"] = total

    print(f"Smoke device: {device}; real dataset subset: 2 demos / {total} steps", flush=True)
    print(f"Outputs: {run_dir}", flush=True)
    commands = []

    def run(label, argv):
        command = [sys.executable, *map(str, argv)]
        commands.append({"label": label, "command": command})
        log = logs / f"{label}.log"
        print(f"Running {label} ...", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=q3, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(errors="replace")[-8000:], flush=True)
            raise RuntimeError(f"{label} failed; full log: {log}")
        print(f"PASS {label}", flush=True)

    common = ["--dataset_path", subset, "--device", device, "--batch_size", 2,
              "--gradient_steps", 3, "--eval_freq", 2, "--save_freq", 2,
              "--log_freq", 1, "--eval_episodes", 1, "--max_episode_steps", 8]
    for name, script, extra in [
        ("bc", "bc.py", []),
        ("diffusion_policy_eps", "diffusion_policy.py", []),
        ("diffusion_policy_x0", "diffusion_policy.py", ["--predict_x0"]),
    ]:
        run(name, [script, *extra, *common, "--results_dir", run_dir / name])

    # Run the supplied plotter against smoke CSVs, keeping formal results separate.
    plot_code = (
        "import importlib.util; "
        "s=importlib.util.spec_from_file_location('plotter','plot_completion_vs_gradient_steps.py'); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.RESULTS_DIR={str(run_dir)!r}; "
        f"m.OUTPUT_PATH={str(run_dir / 'completion_vs_gradient_steps.png')!r}; m.main()"
    )
    run("completion_plot", ["-c", plot_code])
    for mode in ["eps", "x0"]:
        run(f"sampling_{mode}", [
            "plot_diffusion_generation.py",
            *(["--predict_x0"] if mode == "x0" else []),
            "--dataset_path", subset, "--device", device,
            "--results_dir", run_dir / f"diffusion_policy_{mode}",
        ])

    # Load each saved final checkpoint through the ordinary inference entry point.
    for name, script, identifier, extra in [
        ("bc", "bc.py", "final", []),
        ("diffusion_policy_eps", "diffusion_policy.py", "eps_final", []),
        ("diffusion_policy_x0", "diffusion_policy.py", "x0_final", ["--predict_x0"]),
    ]:
        run(f"reload_{name}", [
            script, *extra, "--mode", "inference", "--dataset_path", subset,
            "--device", device, "--eval_episodes", 1, "--max_episode_steps", 8,
            "--model_path", run_dir / name / "models" / f"model_{identifier}.pt",
            "--results_dir", run_dir / "reload" / name,
        ])

    import cv2

    for name in ["bc", "diffusion_policy_eps", "diffusion_policy_x0"]:
        with (run_dir / name / "train_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["step"]) for row in rows] == [1, 2, 3]
        assert all(math.isfinite(float(row["loss"])) for row in rows)
        with (run_dir / name / "eval_metrics.csv").open() as stream:
            evaluations = list(csv.DictReader(stream))
        assert [int(row["step"]) for row in evaluations] == [2, 3]
    image = cv2.imread(str(run_dir / "completion_vs_gradient_steps.png"))
    assert image is not None
    for mode in ["eps", "x0"]:
        video = run_dir / f"diffusion_policy_{mode}" / f"reverse_diffusion_{mode}.mp4"
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert ok and frame is not None and frames > 1

    summary = {"status": "PASS", "device": device, "torch": torch.__version__,
               "demonstrations": 2, "updates_per_model": 3,
               "formal_training_complete": False, "commands": commands}
    (run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"ALL PASS: training, evaluation, checkpoint reload, plot, and both videos.\n{run_dir}", flush=True)
    print("Smoke results validate execution only; complete full Slurm training for Q3.3.", flush=True)


if __name__ == "__main__":
    main()
