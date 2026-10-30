"""Behavior Cloning baseline for the Robomimic "square" image task.

Uses the exact same dataloader (utils/dataset.py's RobomimicImageDataset)
and network backbone (networks/'s DiT1d + MultiImageObsCondition, with
identical hyperparameters) as diffusion_policy.py -- the only difference is
how the shared DiT1d backbone is used: instead of denoising a noisy action
over many diffusion steps, BCPolicy calls it once with a dummy all-zeros
action and a fixed dummy timestep, and reads the output directly as the
predicted action (direct regression, MSE loss).

This works because DiT1d routes all conditioning (timestep + observation
embedding) through one combined embedding used for adaLN-Zero modulation at
every transformer block; the "noisy action" input only ever contributes a
position-dependent term. So even with that input zeroed out, the network's
output is still architecturally forced to come from the real observation
condition, via the same pathway the diffusion model uses (see
networks/dit1d.py's docstring).

Unlike diffusion_policy.py, BC predicts and executes a single action per
observation -- no action chunking. It replans every env step.
"""

import argparse
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


class BCPolicy:
    """Direct-regression policy sharing DDPM's bookkeeping shape (ModuleDict
    of backbone + condition net, an EMA shadow copy, AdamW, save/load in the
    same checkpoint format) for pedagogical symmetry with ddpm.py -- but
    without any noise schedule or sampling loop, since there's nothing to
    denoise. Always predicts a single action (no action chunking); the
    backbone is called with a length-1 dummy sequence."""

    def __init__(self, backbone: nn.Module, nn_condition: nn.Module, action_dim: int,
                 ema_rate: float = 0.995, grad_clip_norm: float = None, optim_params: dict = None,
                 clip_action: bool = True, device="cpu"):
        optim_params = optim_params or {"lr": 2e-4, "weight_decay": 1e-5}
        self.device = device
        self.action_dim = action_dim
        self.ema_rate = ema_rate
        self.grad_clip_norm = grad_clip_norm
        self.clip_action = clip_action

        self.model = nn.ModuleDict({"backbone": backbone.to(device), "condition": nn_condition.to(device)})
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        self.model.train()
        self.model_ema.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1.0 - self.ema_rate)

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "model_ema": self.model_ema.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])

    def loss(self, action_gt: torch.Tensor, condition):
        condition_vec = self.model["condition"](condition)
        b = action_gt.shape[0]
        dummy_x = torch.zeros(b, 1, self.action_dim, device=self.device)
        dummy_t = torch.zeros(b, dtype=torch.long, device=self.device)
        pred = self.model["backbone"](dummy_x, dummy_t, condition_vec)
        return ((pred - action_gt) ** 2).mean()

    def update(self, action_gt, condition, update_ema=True):
        loss = self.loss(action_gt, condition)
        loss.backward()
        grad_norm = (
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) if self.grad_clip_norm else None
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema:
            self.ema_update()
        return {"loss": loss.item(), "grad_norm": grad_norm}

    @torch.no_grad()
    def predict(self, condition, use_ema: bool = True):
        model = self.model_ema if use_ema else self.model
        condition_vec = model["condition"](condition)
        b = next(iter(condition.values())).shape[0]
        dummy_x = torch.zeros(b, 1, self.action_dim, device=self.device)
        dummy_t = torch.zeros(b, dtype=torch.long, device=self.device)
        pred = model["backbone"](dummy_x, dummy_t, condition_vec)
        if self.clip_action:
            pred = pred.clamp(-1.0, 1.0)
        return pred


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
    # observation encoder -- identical to diffusion_policy.py's, so the
    # backbones being compared are genuinely the same architecture
    p.add_argument("--rgb_model", default="resnet18")
    p.add_argument("--crop_shape", type=int, nargs=2, default=(76, 76))
    p.add_argument("--random_crop", type=bool, default=True)
    p.add_argument("--use_group_norm", type=bool, default=True)
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
    p.add_argument("--results_dir", default="results/bc")
    # mode
    p.add_argument("--mode", choices=["train", "inference"], default="train")
    p.add_argument("--model_path", default=None)
    return p


def build_networks(args):
    nn_condition = MultiImageObsCondition(
        shape_meta=SHAPE_META, emb_dim=256, rgb_model_name=args.rgb_model,
        crop_shape=tuple(args.crop_shape), random_crop=args.random_crop,
        use_group_norm=args.use_group_norm)
    backbone = DiT1d(args.action_dim, emb_dim=256, d_model=320, n_heads=10, depth=2)
    return nn_condition, backbone


def evaluate(args, env, dataset, policy, use_ema=True):
    rewards, successes = [], []
    for ep in range(args.eval_episodes):
        obs = env.reset()
        ep_reward, t, done = 0.0, 0, False
        while not done and t < args.max_episode_steps:
            condition = {}
            for k, v in obs.items():
                nobs = dataset.normalizer["obs"][k].normalize(v.astype(np.float32))[None]
                condition[k] = torch.tensor(nobs, device=args.device, dtype=torch.float32)

            naction = policy.predict(condition, use_ema=use_ema)
            naction = naction.detach().cpu().numpy()
            action_pred = dataset.normalizer["action"].unnormalize(naction)
            if args.abs_action:
                action_pred = dataset.undo_transform_action(action_pred)
            action = action_pred[0, 0]

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            t += 1

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
    logger = CSVLogger(args.results_dir)

    # horizon=1, pad_after=0: no action chunking -- each sample is a single
    # (observation, action) pair.
    dataset = RobomimicImageDataset(
        args.dataset_path, shape_meta=SHAPE_META, horizon=1,
        pad_before=0, pad_after=0, abs_action=args.abs_action)
    print(dataset)

    nn_condition, backbone = build_networks(args)
    report_parameters(backbone)
    report_parameters(nn_condition)

    policy = BCPolicy(
        backbone=backbone, nn_condition=nn_condition, action_dim=args.action_dim,
        optim_params={"lr": args.lr}, device=args.device)

    if args.mode == "inference":
        assert args.model_path, "--model_path is required for --mode inference"
        policy.load(args.model_path)
        policy.eval()
        env = make_single_env(args, SHAPE_META)
        evaluate(args, env, dataset, policy)
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

        loss = policy.update(naction, condition)["loss"]
        losses.append(loss)
        n_gradient_step += 1

        if n_gradient_step % args.log_freq == 0:
            logger.log({"step": n_gradient_step, "loss": float(np.mean(losses)),
                        "elapsed": time.time() - start_time}, category="train")
            losses = []

        if n_gradient_step % args.save_freq == 0 and n_gradient_step > 0:
            logger.save_agent(policy, identifier="latest")

        if n_gradient_step % args.eval_freq == 0 and n_gradient_step < args.gradient_steps:
            policy.eval()
            metrics = evaluate(args, env, dataset, policy)
            logger.log({"step": n_gradient_step, **metrics}, category="eval")
            policy.train()

        if n_gradient_step >= args.gradient_steps:
            break

    logger.save_agent(policy, identifier="final")
    policy.eval()
    metrics = evaluate(args, env, dataset, policy)
    logger.log({"step": n_gradient_step, **metrics}, category="eval")


if __name__ == "__main__":
    main()
