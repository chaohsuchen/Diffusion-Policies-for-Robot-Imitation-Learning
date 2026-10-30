"""Everything Q3 needs besides the dataset loader (see dataset.py /
dataset_utils.py), consolidated into one flat file: small numeric helpers,
rotation conversions, the zarr-backed replay buffer + Jpeg2k image codec the
dataset uses, image-crop augmentation, the (simplified) robomimic env
wrapper, and a plain CSV/stdout logger.

Ported from several CleanDiffuser source files (cleandiffuser/utils/utils.py,
cleandiffuser/dataset/{replay_buffer,imagecodecs,rotation_conversions}.py,
cleandiffuser/utils/{crop_randomizer,tensor_utils}.py,
cleandiffuser/env/robomimic/robomimic_image_wrapper.py,
pipelines/{dp_robomimic_image.py,utils.py}), trimmed to only what this repo
actually uses -- see Q3/networks/, Q3/ddpm.py, Q3/diffusion_policy.py and
Q3/bc.py for the call sites.
"""

import collections
import csv
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf

import gym
from gym import spaces
from numcodecs.abc import Codec as _NumcodecsCodec

if "MUJOCO_GL" not in os.environ:
    # Pick a headless-friendly OpenGL backend for mujoco/robosuite's
    # offscreen renderer: EGL on machines with an NVIDIA GPU, OSMesa
    # (software) otherwise. Same auto-detection Q2 uses (see
    # Q2/ballcatch_env/gym_thing/__init__.py) -- both work without a
    # GLX-capable display, unlike the default GLFW backend.
    def _has_nvidia_gpu():
        return shutil.which("nvidia-smi") is not None and subprocess.run(
            ["nvidia-smi"], capture_output=True
        ).returncode == 0

    if sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ["MUJOCO_GL"] = "egl" if _has_nvidia_gpu() else "osmesa"


# =============================================================================
# Small numeric helpers (cleandiffuser/utils/utils.py)
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def at_least_ndim(x, ndim: int, pad: int = 0):
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            pad_shape = (1,) * (ndim - x.ndim)
            return np.reshape(x, x.shape + pad_shape if pad == 0 else pad_shape + x.shape)
        return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            pad_shape = (1,) * (ndim - x.ndim)
            return torch.reshape(x, x.shape + pad_shape if pad == 0 else pad_shape + x.shape)
        return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (np.ndarray, list, tuple, int, float)):
        return torch.tensor(x, device=device)
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02, T: int = 1000):
    return np.linspace(beta_min, beta_max, T)


def cosine_beta_schedule(s: float = 0.008, T: int = 1000):
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clip(None, 0.999)

def sigmoid_beta_schedule(T: int = 1000):
    if T < 1:
        raise ValueError("T must be at least 1")

    progress = np.linspace(
        0.0, 1.0, T + 1, dtype=np.float64
    )
    # 6 and 3 are hyperparam, 6 decides the slope 3 puts the center at 0.5
    # This is a sigmoid curve
    curve = 1.0 / (
        1.0 + np.exp(-(6.0 * progress - 3.0))
    )

    alpha_bar = (
        (curve[-1] - curve)
        / (curve[-1] - curve[0])
    )

    # element wise division.
    beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]

    return np.clip(beta, 1e-8, 0.999)


def _to_str(num):
    return f"{(num / 1e6):.2f} M" if num >= 1e6 else f"{(num / 1e3):.2f} k"


def report_parameters(model: nn.Module):
    counts = {k: p.numel() for k, p in model.named_parameters() if p.requires_grad}
    n_parameters = sum(counts.values())
    print(f"Total parameters: {_to_str(n_parameters)}")
    return n_parameters


# =============================================================================
# Rotation conversions (cleandiffuser/dataset/rotation_conversions.py, itself
# adapted from pytorch3d). Only the axis_angle <-> matrix <-> rotation_6d
# chain is ported since that's the only pair RotationTransformer is used
# for (converting robomimic's abs_action axis-angle rotations to 6D).
# =============================================================================

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(torch.stack(
        [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ], dim=-2)

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    import torch.nn.functional as F
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


# =============================================================================
# Trimmed tensor_utils (cleandiffuser/utils/tensor_utils.py). Only the plain
# single-tensor operations CropRandomizer actually calls -- the original
# versions are generic nested-dict/list/tuple appliers, which we don't need
# since these are only ever called on bare tensors here.
# =============================================================================

def flatten(x: torch.Tensor, begin_axis: int = 1) -> torch.Tensor:
    fixed_size = x.size()[:begin_axis]
    return x.reshape(*list(fixed_size), -1)


def reshape_dimensions(x: torch.Tensor, begin_axis: int, end_axis: int, target_dims) -> torch.Tensor:
    assert begin_axis <= end_axis >= 0
    assert end_axis < len(x.shape)
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)


def join_dimensions(x: torch.Tensor, begin_axis: int, end_axis: int) -> torch.Tensor:
    return reshape_dimensions(x, begin_axis, end_axis, target_dims=[-1])


def unsqueeze_expand_at(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    x = x.unsqueeze(dim=dim)
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)


# =============================================================================
# CropRandomizer (cleandiffuser/utils/crop_randomizer.py), verbatim aside
# from the tensor_utils import above.
# =============================================================================

class CropRandomizer(nn.Module):
    """Randomly sample crops at input, and average across crop features at output."""

    def __init__(self, input_shape, crop_height, crop_width, num_crops=1, pos_enc=False):
        super().__init__()
        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]
        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def forward_in(self, inputs):
        assert len(inputs.shape) >= 3
        if self.training:
            out, _ = sample_random_image_crops(
                images=inputs, crop_height=self.crop_height, crop_width=self.crop_width,
                num_crops=self.num_crops, pos_enc=self.pos_enc)
            return join_dimensions(out, 0, 1)
        else:
            out = ttf.center_crop(img=inputs, output_size=(self.crop_height, self.crop_width))
            if self.num_crops > 1:
                B, C, H, W = out.shape
                out = out.unsqueeze(1).expand(B, self.num_crops, C, H, W).reshape(-1, C, H, W)
            return out

    def forward(self, inputs):
        return self.forward_in(inputs)

    def __repr__(self):
        return (f"{self.__class__.__name__}(input_shape={self.input_shape}, "
                f"crop_size=[{self.crop_height}, {self.crop_width}], num_crops={self.num_crops})")


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (ndim_im_shape == ndim_indices_shape + 2)

    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_w - crop_width)).all().item()

    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    crop_in_grid = torch.cat((crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1)

    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(grid_reshape)

    all_crop_inds = all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1]
    all_crop_inds = unsqueeze_expand_at(all_crop_inds, size=image_c, dim=-3)
    all_crop_inds = flatten(all_crop_inds, begin_axis=-2)

    images_to_crop = unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    reshape_axis = len(crops.shape) - 1
    crops = reshape_dimensions(crops, begin_axis=reshape_axis, end_axis=reshape_axis, target_dims=(crop_height, crop_width))

    if is_padded:
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(images, crop_height, crop_width, num_crops, pos_enc=False):
    device = images.device
    source_im = images
    if pos_enc:
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x))
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)
        source_im = torch.cat((source_im, position_enc), dim=-3)

    image_c, image_h, image_w = source_im.shape[-3:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width

    crop_inds_h = (max_sample_h * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds_w = (max_sample_w * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds = torch.cat((crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1)

    crops = crop_image_from_indices(
        images=source_im, crop_indices=crop_inds, crop_height=crop_height, crop_width=crop_width)
    return crops, crop_inds


# =============================================================================
# ReplayBuffer (cleandiffuser/dataset/replay_buffer.py), trimmed to the
# dict-like read API + add_episode -- the save/copy/load classmethods aren't
# used since RobomimicImageDataset builds the zarr group itself and wraps it
# directly with ReplayBuffer(root).
# =============================================================================

class ReplayBuffer:
    """Zarr-based temporal datastructure. Assumes first dimension is time."""

    def __init__(self, root):
        assert "data" in root
        assert "meta" in root
        assert "episode_ends" in root["meta"]
        for key, value in root["data"].items():
            assert value.shape[0] == root["meta"]["episode_ends"][-1]
        self.root = root

    @property
    def data(self):
        return self.root["data"]

    @property
    def meta(self):
        return self.root["meta"]

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    @property
    def n_steps(self):
        return 0 if len(self.episode_ends) == 0 else self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data


# =============================================================================
# Jpeg2k image codec (cleandiffuser/dataset/imagecodecs.py), trimmed to just
# the one codec RobomimicImageDataset registers/uses.
# =============================================================================

def _protective_squeeze(x):
    """Squeeze leading dims only if they're not the last 3 (image) dims."""
    img_shape = x.shape[-3:]
    if len(x.shape) > 3:
        n_imgs = np.prod(x.shape[:-3])
        if n_imgs > 1:
            img_shape = (-1,) + img_shape
    return x.reshape(img_shape)


class Jpeg2k(_NumcodecsCodec):
    """JPEG 2000 codec for numcodecs. Constructed directly by
    dataset.py's _convert_robomimic_to_replay to compress rgb frames in the
    zarr replay buffer; register_codecs() below makes it loadable by id from
    that buffer afterwards. Relies on numcodecs.abc.Codec's default
    get_config/from_config (reflects self.__dict__), same as upstream."""

    codec_id = "imagecodecs_jpeg2k"

    def __init__(self, level=None, codecformat=None, colorspace=None, tile=None, reversible=None,
                 bitspersample=None, resolutions=None, numthreads=None, verbose=0):
        self.level = level
        self.codecformat = codecformat
        self.colorspace = colorspace
        self.tile = None if tile is None else tuple(tile)
        self.reversible = reversible
        self.bitspersample = bitspersample
        self.resolutions = resolutions
        self.numthreads = numthreads
        self.verbose = verbose

    def encode(self, buf):
        import imagecodecs
        buf = _protective_squeeze(np.asarray(buf))
        return imagecodecs.jpeg2k_encode(
            buf, level=self.level, codecformat=self.codecformat, colorspace=self.colorspace, tile=self.tile,
            reversible=self.reversible, bitspersample=self.bitspersample, resolutions=self.resolutions,
            numthreads=self.numthreads, verbose=self.verbose)

    def decode(self, buf, out=None):
        import imagecodecs
        return imagecodecs.jpeg2k_decode(buf, verbose=self.verbose, numthreads=self.numthreads, out=out)


def register_codecs():
    """Register Jpeg2k with numcodecs so zarr can load it by codec id alone
    (needed when reading arrays back out of the replay buffer)."""
    from numcodecs.registry import get_codec, register_codec

    # numcodecs.abc.Codec requires get_config/from_config for full compliance,
    # but zarr only calls encode/decode on already-constructed instances
    # stored inline in each array's metadata, so the minimal class above is
    # sufficient here.
    try:
        get_codec({"id": Jpeg2k.codec_id})
    except ValueError:
        register_codec(Jpeg2k)


# =============================================================================
# Robomimic env wrapper (cleandiffuser/env/robomimic/robomimic_image_wrapper.py)
# + a simplified single-in-process-env receding-horizon wrapper (adapted from
# cleandiffuser/env/wrapper.py's MultiStepWrapper, dropping the AsyncVectorEnv
# / dill cross-process plumbing that a single env doesn't need) + a
# make_single_env() helper mirroring pipelines/dp_robomimic_image.py's
# make_async_envs() for exactly one env, no video recording.
# =============================================================================

class RobomimicImageWrapper(gym.Env):
    def __init__(self, env, shape_meta: dict, init_state=None, render_obs_key="agentview_image"):
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False

        action_shape = shape_meta["action"]["shape"]
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("image"):
                min_value, max_value = 0, 1
            observation_space[key] = spaces.Box(low=min_value, high=max_value, shape=shape, dtype=np.float32)
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        self.render_cache = raw_obs[self.render_obs_key]
        return {key: raw_obs[key] for key in self.observation_space.keys()}

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                self.seed_state_map[seed] = self.env.get_state()["states"]
            self._seed = None
        else:
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        return self.get_observation(raw_obs), reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        return (img * 255).astype(np.uint8)


def make_single_env(args, shape_meta: dict):
    """Build one synchronous robomimic-image env, matching
    pipelines/dp_robomimic_image.py's make_async_envs() for exactly one env
    with no video recording."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    dataset_path = os.path.expanduser(args.dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["use_object_obs"] = False
    if args.abs_action:
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)
    env.env.hard_reset = False

    return RobomimicImageWrapper(env=env, shape_meta=shape_meta, render_obs_key=args.render_obs_key)


# =============================================================================
# Plain CSV/stdout logger, replacing pipelines/utils.py's wandb-based Logger.
# =============================================================================

class CSVLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.model_dir = os.path.join(log_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self._fieldnames = {}  # one schema per category (train/eval have different fields)

    def log(self, d: dict, category: str):
        assert "step" in d
        print(f"[{category}][{d['step']}]", " / ".join(f"{k} {v:.4f}" for k, v in d.items() if k != "step"))
        csv_path = os.path.join(self.log_dir, f"{category}_metrics.csv")
        write_header = category not in self._fieldnames
        if write_header:
            self._fieldnames[category] = list(d.keys())
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames[category])
            if write_header:
                writer.writeheader()
            writer.writerow(d)

    def save_agent(self, agent, identifier="final"):
        path = os.path.join(self.model_dir, f"model_{identifier}.pt")
        agent.save(path)
        print(f"model_{identifier} saved to {path}")
