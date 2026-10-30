"""Sequence sampling, rotation conversion, and normalization helpers used by
RobomimicImageDataset (see dataset.py). Ported near-verbatim from
CleanDiffuser's cleandiffuser/dataset/dataset_utils.py, dropping
CDFNormalizer/GaussianNormalizer/empirical_cdf and loop_two_dataloaders
(unused D4RL-era leftovers).
"""

from typing import Callable, Dict, Union

import numba
import numpy as np
import torch

import utils.common as rc
from utils.common import ReplayBuffer


# -----------------------------------------------------------------------------
# SequenceSampler
#
# Observation Horizon: To | n_obs_steps
# Action Horizon: Ta | n_action_steps
# Prediction Horizon: T | horizon
# To = 3
# Ta = 4
# T = 6
# |o|o|o|
# | | |a|a|a|a|
# pad_before = 2
# pad_after = 3
# -----------------------------------------------------------------------------

@numba.jit(nopython=True)
def create_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0, pad_after: int = 0,
        debug: bool = True) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


class SequenceSampler:
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            sequence_length: int,
            pad_before: int = 0,
            pad_after: int = 0,
            keys=None,
            key_first_k=dict(),
            zero_padding: bool = False,
    ):
        """key_first_k: dict str -> int. Only take the first k timesteps from
        these keys (avoids allocating/loading full-horizon image sequences
        when only the first n_obs_steps are actually used as conditioning)."""
        super().__init__()
        assert sequence_length >= 1

        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        indices = create_indices(
            episode_ends=episode_ends, sequence_length=sequence_length,
            pad_before=pad_before, pad_after=pad_after)

        self.indices = indices
        self.keys = list(keys)
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.zero_padding = zero_padding
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], fill_value=np.nan, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(shape=(self.sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


# -----------------------------------------------------------------------------
# Rotation transformer (rotation functions themselves live in utils/common.py)
# -----------------------------------------------------------------------------

class RotationTransformer:
    valid_reps = ["axis_angle", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """Only axis_angle <-> rotation_6d (via matrix) is supported -- the
        only pair this repo's abs_action conversion needs."""
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(rc, f"{from_rep}_to_matrix"), getattr(rc, f"matrix_to_{from_rep}")]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(rc, f"matrix_to_{to_rep}"), getattr(rc, f"{to_rep}_to_matrix")]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.tensor(x)
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x):
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x):
        return self._apply_funcs(x, self.inverse_funcs)


# -----------------------------------------------------------------------------
# Normalizers
# -----------------------------------------------------------------------------

class ImageNormalizer:
    """Normalizes image data from range [0, 1] to [-1, 1]."""

    def normalize(self, x):
        return x * 2.0 - 1.0

    def unnormalize(self, x):
        return (x + 1.0) / 2.0


class MinMaxNormalizer:
    """Normalizes data to [-1, 1] via per-feature min/max expansion."""

    def __init__(self, X):
        X = X.reshape(-1, X.shape[-1]).astype(np.float32)
        self.min, self.max = np.min(X, axis=0), np.max(X, axis=0)
        self.range = self.max - self.min
        if np.any(self.range == 0):
            print("Warning: Some features have the same min and max value. These will be set to 0.")
            self.range[self.range == 0] = 1

    def normalize(self, x):
        x = x.astype(np.float32)
        nx = (x - self.min) / self.range
        return nx * 2 - 1

    def unnormalize(self, x):
        x = x.astype(np.float32)
        nx = (x + 1) / 2
        return nx * self.range + self.min


class EmptyNormalizer:
    """Does nothing."""

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


def loop_dataloader(dl):
    while True:
        for b in dl:
            yield b
