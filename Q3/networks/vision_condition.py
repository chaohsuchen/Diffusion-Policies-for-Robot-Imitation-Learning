"""MultiImageObsCondition: the vision + low-dim observation encoder shared by
diffusion_policy.py and bc.py.

Ported near-verbatim from CleanDiffuser's
cleandiffuser/nn_condition/multi_image_condition.py, dropping the
classifier-free-guidance `mask` argument (unused here -- condition is always
fully present) and the BaseNNCondition ABC (only one condition class exists
in this repo, so the abstraction adds nothing).
"""

import copy
from typing import Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision

from utils.common import CropRandomizer


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """predicate: return True if the module should be replaced. func: return
    the replacement module."""
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    bn_list = [
        k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def get_resnet(name, weights=None, **kwargs):
    """name: resnet18, resnet34, resnet50. weights: "IMAGENET1K_V1" or None."""
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


class MultiImageObsCondition(nn.Module):
    """
    Input:
        - condition: {"key1": (b, *shape1), ...} with rgb keys shaped
          (b, C, H, W) and low_dim keys shaped (b, D) -- always a single
          current observation, no history stacking.
    Output:
        - condition: (b, emb_dim), matching DiT1d's expected condition shape.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str,
        emb_dim: int = 256,
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
    ):
        super().__init__()
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        if "resnet" in rgb_model_name:
            rgb_model = get_resnet(rgb_model_name)
        else:
            raise ValueError("Only resnet backbones are supported")

        if share_rgb_model:
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if obs_type == "rgb":
                rgb_keys.append(key)
                this_model = None
                if not share_rgb_model:
                    this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
                        )
                    key_model_map[key] = this_model

                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    h, w = resize_shape[key] if isinstance(resize_shape, dict) else resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    h, w = crop_shape[key] if isinstance(crop_shape, dict) else crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape, crop_height=h, crop_width=w, num_crops=1, pos_enc=False)
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))

                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                key_transform_map[key] = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
            elif obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape(), emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

    def multi_image_forward(self, obs_dict):
        batch_size = None
        features = list()

        if self.share_rgb_model:
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                batch_size = img.shape[0] if batch_size is None else batch_size
                assert img.shape[1:] == self.key_shape_map[key]
                imgs.append(self.key_transform_map[key](img))
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map["rgb"](imgs)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                batch_size = img.shape[0] if batch_size is None else batch_size
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                features.append(self.key_model_map[key](img))

        for key in self.low_dim_keys:
            data = obs_dict[key]
            batch_size = data.shape[0] if batch_size is None else batch_size
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        return torch.cat(features, dim=-1)

    def forward(self, obs_dict):
        features = self.multi_image_forward(obs_dict)
        return self.mlp(features)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        for key, attr in self.shape_meta["obs"].items():
            shape = tuple(attr["shape"])
            example_obs_dict[key] = torch.zeros((1,) + shape, dtype=self.dtype, device=self.device)
        example_output = self.multi_image_forward(example_obs_dict)
        return example_output.shape[1:][0]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
