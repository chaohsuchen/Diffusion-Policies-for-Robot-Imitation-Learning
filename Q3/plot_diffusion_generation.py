"""Visualizes DDPM's reverse-diffusion process for a whole action chunk.

Loads a trained diffusion_policy.py checkpoint, conditions on one
deterministic dataset sample (no shuffling), runs DDPM.sample() with
record_sampling=True, and renders a video: each frame is a 3D quiver plot of
every action's movement vector (delta pos x/y/z) in the chunk, one frame per
diffusion step, played in reverse-diffusion order (T -> 0) so you can watch
the chunk denoise from Gaussian noise into a coherent trajectory.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ddpm import DDPM
from diffusion_policy import SHAPE_META, build_arg_parser, build_networks
from utils.common import set_seed
from utils.dataset import RobomimicImageDataset


def main():
    p = build_arg_parser()
    p.add_argument("--sample_idx", type=int, default=0,
                    help="Dataset index to condition on (deterministic, bypasses the shuffled DataLoader).")
    p.add_argument("--output", default=None,
                    help="Where to save the video. Defaults to results_dir/reverse_diffusion_<mode>.mp4.")
    p.add_argument("--fps", type=int, default=12, help="Video framerate.")
    p.add_argument("--hold_frames", type=int, default=3,
                    help="How many video frames to hold each diffusion step for (pacing).")
    args = p.parse_args()
    set_seed(args.seed)

    mode_tag = "x0" if args.predict_x0 else "eps"
    if args.results_dir is None:
        suffix = (
            ""
            if args.beta_schedule == "cosine"
            else f"_{args.beta_schedule}"
        )
        args.results_dir = (
            f"results/diffusion_policy_{mode_tag}{suffix}"
        )
    model_path = args.model_path or os.path.join(args.results_dir, "models", f"model_{mode_tag}_final.pt")
    output_path = args.output or os.path.join(args.results_dir, f"reverse_diffusion_{mode_tag}.mp4")

    dataset = RobomimicImageDataset(
        args.dataset_path, shape_meta=SHAPE_META, horizon=args.horizon,
        pad_before=0, pad_after=args.action_steps - 1, abs_action=args.abs_action)
    print(dataset)

    nn_condition, nn_diffusion = build_networks(args)
    agent = DDPM(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        diffusion_steps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
        predict_noise=not args.predict_x0,
        device=args.device,
    )
    agent.load(model_path)
    agent.eval()
    print(f"Loaded {mode_tag}-prediction checkpoint from {model_path}")

    # dataset[i]'s obs are already normalized (see RobomimicImageDataset.__getitem__)
    # -- just add a batch dim to use directly as the sampler's condition.
    sample = dataset[args.sample_idx]
    condition = {k: v.unsqueeze(0).to(args.device) for k, v in sample["obs"].items()}

    prior = torch.zeros((1, args.horizon, args.action_dim), device=args.device)
    _, sample_record = agent.sample(
        prior=prior, n_samples=1, sample_steps=args.diffusion_steps,
        use_ema=True, temperature=args.temperature, condition=condition,
        record_sampling=True)

    # sample_record[0] is pure noise x_T, sample_record[-1] is the final clean
    # x_0 -- already in reverse-diffusion (T -> 0) order. Each action's first
    # 3 dims are its delta pos(x, y, z). Actions here are relative (delta),
    # not absolute (args.abs_action=False by default), so chaining them
    # tip-to-tail from the origin traces the predicted end-effector path.
    movement = torch.stack([xt[0, :, :3] for xt in sample_record], dim=0).cpu().numpy()  # (T+1, horizon, 3)
    diffusion_steps_axis = list(range(args.diffusion_steps, -1, -1))

    horizon = args.horizon
    origin = np.zeros((movement.shape[0], 1, 3))
    path = np.concatenate([origin, np.cumsum(movement, axis=1)], axis=1)  # (T+1, horizon+1, 3)

    segment_colors = plt.cm.viridis(np.linspace(0, 1, horizon))  # one per action/segment
    point_colors = plt.cm.viridis(np.linspace(0, 1, horizon + 1))  # one per path vertex

    dpi = 100
    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, horizon - 1))
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label="action index in chunk")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, args.fps, (width, height))

    for i, t in enumerate(diffusion_steps_axis):
        ax.cla()
        points = path[i]  # (horizon + 1, 3), points[0] is the origin/start
        segments = np.stack([points[:-1], points[1:]], axis=1)  # (horizon, 2, 3)
        ax.add_collection3d(Line3DCollection(segments, colors=segment_colors, linewidth=2))
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=point_colors, s=20)
        ax.scatter([0], [0], [0], color="black", marker="x", s=80, label="start (end-effector origin)")

        # Auto-scale per frame (with equal aspect) rather than a fixed range
        # shared across the whole video: the path's extent shrinks by orders
        # of magnitude from the noisy early steps to the converged final
        # steps, and a single fixed scale would make the converged path
        # invisible. The axis range is annotated in the title since it
        # changes frame to frame.
        frame_lim = max(float(np.abs(points).max()) * 1.2, 0.05)
        ax.set_xlim(-frame_lim, frame_lim)
        ax.set_ylim(-frame_lim, frame_lim)
        ax.set_zlim(-frame_lim, frame_lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x (from start)")
        ax.set_ylabel("y (from start)")
        ax.set_zlabel("z (from start)")
        ax.legend(loc="upper left")
        ax.set_title(f"Reverse diffusion of the predicted end-effector path ({mode_tag}-prediction)\n"
                      f"diffusion step t = {t}  (axis range: ±{frame_lim:.2f})")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        for _ in range(args.hold_frames):
            writer.write(frame_bgr)

    writer.release()
    plt.close(fig)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
