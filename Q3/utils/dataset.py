"""RobomimicImageDataset, ported close to verbatim from CleanDiffuser's
cleandiffuser/dataset/robomimic_dataset.py (per the assignment's "keep their
dataloader mostly the same" requirement). Dropped: the sibling low-dim-only
RobomimicDataset class in the same source file (unused -- this repo only
supports the image variant).
"""

import concurrent.futures
import os
from collections import defaultdict
from typing import Dict

import h5py
import numpy as np
import torch
import zarr
from tqdm import tqdm

from utils.common import ReplayBuffer, Jpeg2k, register_codecs
from utils.dataset_utils import (
    SequenceSampler, RotationTransformer, dict_apply, MinMaxNormalizer, ImageNormalizer,
)

register_codecs()


class RobomimicImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
        rotation_rep="rotation_6d",
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep=rotation_rep)

        self.replay_buffer = _convert_robomimic_to_replay(
            store=zarr.MemoryStore(), shape_meta=shape_meta, dataset_path=dataset_dir,
            abs_action=abs_action, rotation_transformer=self.rotation_transformer)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        # Only condition on the current observation (no history stacking), so
        # only the first timestep of each obs key needs to be loaded per
        # sample -- avoids allocating full-horizon image sequences.
        key_first_k = {key: 1 for key in rgb_keys + lowdim_keys}
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=horizon,
            pad_before=pad_before, pad_after=pad_after, key_first_k=key_first_k)

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])
        return normalizer

    def __str__(self) -> str:
        return f"Keys: {list(self.replay_buffer.keys())} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # Current-observation-only conditioning: take just the first (i.e.
        # "now") timestep of each obs key, dropping the time dimension.
        obs_dict = dict()
        for key in self.rgb_keys:
            # H,W,C -> C,H,W ; uint8 -> float32/255
            obs_dict[key] = np.moveaxis(sample[key][0], -1, 0).astype(np.float32) / 255.0
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][0].astype(np.float32)
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        return {"obs": dict_apply(obs_dict, torch.tensor), "action": torch.tensor(action)}

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)
        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)
        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True
        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer,
                                  n_workers=None, max_inflight_tasks=None):
    """Convert a robomimic hdf5 dataset (data/demo_N/{obs/<key>,actions}) to
    an in-memory zarr-backed ReplayBuffer, Jpeg2k-compressing rgb frames."""
    if n_workers is None:
        # Respect the job allocation; use at most four threads on a local Mac.
        n_workers = max(1, int(os.environ.get(
            "SLURM_CPUS_PER_TASK", min(4, os.cpu_count() or 1))))
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys, lowdim_keys = list(), list()
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        demos = file["data"]
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        meta_group.array("episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True)

        # lowdim data + action (with abs-action rotation conversion)
        for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data, abs_action=abs_action, rotation_transformer=rotation_transformer)
                assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta["obs"][key]["shape"])
            data_group.array(
                name=key, data=this_data, shape=this_data.shape, chunks=this_data.shape,
                compressor=None, dtype=this_data.dtype)

        # rgb data (Jpeg2k-compressed, threaded copy from hdf5 -> zarr)
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False

        with tqdm(total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    c, h, w = tuple(shape_meta["obs"][key]["shape"])
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key, shape=(n_steps, h, w, c), chunks=(1, h, w, c),
                        compressor=this_compressor, dtype=np.uint8)
                    for episode_idx in range(len(demos)):
                        demo = demos[f"demo_{episode_idx}"]
                        hdf5_arr = demo["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                completed, futures = concurrent.futures.wait(
                                    futures, return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))
                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

    return ReplayBuffer(root)
