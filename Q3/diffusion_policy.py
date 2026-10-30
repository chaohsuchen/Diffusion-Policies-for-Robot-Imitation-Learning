"""Diffusion Policy: a DDPM-based visuomotor policy for the Robomimic
"square" image task.

This is the flat replacement for CleanDiffuser's Hydra-driven
pipelines/dp_robomimic_image.py -- same dataset, network, and DDPM math, but
argparse instead of Hydra, one env instead of an AsyncVectorEnv pool, plain
CSV/stdout logging instead of wandb, and no video recording.

See bc.py for the explicit-regression baseline, which shares this script's
dataloader (utils/dataset.py) and network backbone (networks/) exactly.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ddpm import DDPM
from networks import DiT1d, MultiImageObsCondition
from utils.common import CSVLogger, make_single_env, report_parameters, set_seed
from utils.dataset import RobomimicImageDataset
from utils.dataset_utils import loop_dataloader

SHAPE_META = {
    "action": {"shape": [7]},  # delta pos(3) + delta axis-angle rot(3) + gripper(1)
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
        "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
    },
}


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    # env / task
    p.add_argument("--env_name", default="square")
    p.add_argument("--env_type", default="ph")
    p.add_argument("--abs_action", type=bool, default=False)
    p.add_argument("--dataset_path", default="data/robomimic/datasets/square/ph/image.hdf5")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--render_obs_key", default="agentview_image")
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_steps", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    # observation encoder
    p.add_argument("--rgb_model", default="resnet18")
    p.add_argument("--crop_shape", type=int, nargs=2, default=(76, 76))
    p.add_argument("--random_crop", type=bool, default=True)
    p.add_argument("--use_group_norm", type=bool, default=True)
    # diffusion
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument(
        "--beta_schedule",
        choices=["cosine", "linear", "sigmoid"],
        default="cosine",
        help="Noise schedule used for training and sampling.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--predict_x0", action="store_true",
                    help="Predict the clean action x0 directly instead of the noise epsilon.")
    # training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gradient_steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    p.add_argument("--device", default=default_device)
    # eval / logging
    p.add_argument("--eval_episodes", type=int, default=25)
    p.add_argument("--eval_freq", type=int, default=10000)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--save_freq", type=int, default=500)
    p.add_argument("--results_dir", default=None,
                    help="Defaults to results/diffusion_policy_x0 or results/diffusion_policy_eps "
                         "depending on --predict_x0.")
    # mode
    p.add_argument("--mode", choices=["train", "inference"], default="train")
    p.add_argument("--model_path", default=None)
    return p


def build_networks(args):
    nn_condition = MultiImageObsCondition(
        shape_meta=SHAPE_META, emb_dim=256, rgb_model_name=args.rgb_model,
        crop_shape=tuple(args.crop_shape), random_crop=args.random_crop,
        use_group_norm=args.use_group_norm)
    nn_diffusion = DiT1d(args.action_dim, emb_dim=256, d_model=320, n_heads=10, depth=2)
    return nn_condition, nn_diffusion


def evaluate(args, env, dataset, agent, use_ema=True):
    if not 1 <= args.action_steps <= args.horizon:
        raise ValueError("action_steps must be between 1 and horizon")
    rewards, successes = [], []
    for ep in range(args.eval_episodes):
        obs = env.reset()
        ep_reward, t, done = 0.0, 0, False
        while not done and t < args.max_episode_steps:
            condition = {}
            for k, v in obs.items():
                nobs = dataset.normalizer["obs"][k].normalize(v.astype(np.float32))[None]
                condition[k] = torch.tensor(nobs, device=args.device, dtype=torch.float32)

            prior = torch.zeros((1, args.horizon, args.action_dim), device=args.device)
            naction = agent.sample(
                prior=prior, n_samples=1, sample_steps=args.diffusion_steps,
                use_ema=use_ema, temperature=args.temperature, condition=condition)
            naction = naction.detach().cpu().numpy()
            action_pred = dataset.normalizer["action"].unnormalize(naction)

            if args.abs_action:
                action_pred = dataset.undo_transform_action(action_pred)
            action_chunk = action_pred[0]

            # Execute each action once, then replan from the latest observation.
            chunk_reward = 0.0
            for act in action_chunk[:args.action_steps]:
                obs, reward, done, info = env.step(act)
                chunk_reward = max(chunk_reward, float(reward))
                t += 1
                if done or t >= args.max_episode_steps:
                    break

            # Count the chunk maximum once, including terminal success rewards.
            ep_reward += chunk_reward


        rewards.append(ep_reward)
        successes.append(1.0 if ep_reward > 0 else 0.0)
        print(f"[Episode {ep + 1}/{args.eval_episodes}] reward: {ep_reward:.2f} success: {successes[-1]:.0f}")

    mean_reward, mean_success = float(np.mean(rewards)), float(np.mean(successes))
    print(f"Mean reward: {mean_reward:.3f} | Mean success: {mean_success:.3f}")
    return {"mean_reward": mean_reward, "mean_success": mean_success}


def main():
    args = build_arg_parser().parse_args()
    print(f"Using device: {args.device}", flush=True)
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
    logger = CSVLogger(args.results_dir)

    dataset = RobomimicImageDataset(
        args.dataset_path, shape_meta=SHAPE_META, horizon=args.horizon,
        pad_before=0, pad_after=args.action_steps - 1, abs_action=args.abs_action)
    print(dataset)

    nn_condition, nn_diffusion = build_networks(args)
    report_parameters(nn_diffusion)
    report_parameters(nn_condition)

    agent = DDPM(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition, diffusion_steps=args.diffusion_steps,
        beta_schedule=args.beta_schedule, predict_noise=not args.predict_x0, optim_params={"lr": args.lr}, device=args.device)

    if args.mode == "inference":
        assert args.model_path, "--model_path is required for --mode inference"
        agent.load(args.model_path)
        agent.eval()
        env = make_single_env(args, SHAPE_META)
        evaluate(args, env, dataset, agent)
        return

    # Mac uses one process to avoid duplicating the in-memory image dataset.
    # Linux workers stay within the CPUs allocated by Slurm.
    workers = 0 if sys.platform == "darwin" else min(
        4, max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=workers,
        shuffle=True, pin_memory=str(args.device).startswith("cuda"))
    env = make_single_env(args, SHAPE_META)

    n_gradient_step = 0
    losses = []
    start_time = time.time()
    for batch in loop_dataloader(dataloader):
        condition = {k: v.to(args.device) for k, v in batch["obs"].items()}
        naction = batch["action"].to(args.device)

        loss = agent.update(naction, condition)["loss"]
        losses.append(loss)
        n_gradient_step += 1

        if n_gradient_step % args.log_freq == 0:
            logger.log({"step": n_gradient_step, "loss": float(np.mean(losses)),
                        "elapsed": time.time() - start_time}, category="train")
            losses = []

        if n_gradient_step % args.save_freq == 0 and n_gradient_step > 0:
            logger.save_agent(agent, identifier=f"{mode_tag}_latest")

        if n_gradient_step % args.eval_freq == 0 and n_gradient_step < args.gradient_steps:
            agent.eval()
            metrics = evaluate(args, env, dataset, agent)
            logger.log({"step": n_gradient_step, **metrics}, category="eval")
            agent.train()

        if n_gradient_step >= args.gradient_steps:
            break

    logger.save_agent(agent, identifier=f"{mode_tag}_final")
    agent.eval()
    metrics = evaluate(args, env, dataset, agent)
    logger.log({"step": n_gradient_step, **metrics}, category="eval")


if __name__ == "__main__":
    main()
