"""DDPM: a flat, discrete-time-only diffusion model, merging CleanDiffuser's
generic DiffusionModel base class (cleandiffuser/diffusion/basic.py --
optimizer/EMA/save/load bookkeeping) and its DDPM subclass
(cleandiffuser/diffusion/ddpm.py -- noise schedule, training loss, ancestral
sampling) into one class.

DDPM in CleanDiffuser was already fully self-contained and discrete-time-only
(it does not inherit from the separate continuous/discrete SDE class family
in diffusionsde.py), so this is pure trimming, not restructuring. Dropped:
classifier guidance, classifier-free-guidance interpolation, fix_mask/
loss_weight inpainting-style masking, and the sample_x() extra-refinement
variant (all unused here -- condition is always fully present, and there's no
partial-sequence inpainting or extra refinement). Kept the `predict_noise`
toggle (epsilon vs. x0 prediction target -- both are standard equivalent
DDPM parameterizations, useful to compare side by side) and a flat
clip(-1, 1) on samples each denoising step (actions are min-max normalized to
[-1, 1] by the dataset) instead of the parameterized x_max/x_min mechanism.
"""

import warnings
from copy import deepcopy

import torch
import torch.nn as nn

from utils.common import at_least_ndim, cosine_beta_schedule, linear_beta_schedule, sigmoid_beta_schedule
import torch.nn.functional as F


class DDPM:
    def __init__(
        self,
        nn_diffusion: nn.Module,
        nn_condition: nn.Module,
        diffusion_steps: int = 50,
        ema_rate: float = 0.995,
        grad_clip_norm: float = None,
        optim_params: dict = None,
        beta_schedule: str = "cosine",
        beta_schedule_params: dict = None,
        clip_action: bool = True,
        predict_noise: bool = True,
        device="cpu",
    ):
        optim_params = optim_params or {"lr": 2e-4, "weight_decay": 1e-5}
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.diffusion_steps = diffusion_steps
        self.ema_rate = ema_rate
        self.clip_action = clip_action
        self.predict_noise = predict_noise

        self.model = nn.ModuleDict({"diffusion": nn_diffusion.to(device), "condition": nn_condition.to(device)})
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        self.model.train()
        self.model_ema.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)

        beta_schedule_params = dict(beta_schedule_params or {})
        beta_schedule_params["T"] = diffusion_steps
        schedules = {
            "linear": linear_beta_schedule,
            "cosine": cosine_beta_schedule,
            "sigmoid": sigmoid_beta_schedule,
        }

        if beta_schedule not in schedules:
            raise ValueError(
                f"Unknown beta schedule: {beta_schedule}"
            )

        beta = schedules[beta_schedule](
            **beta_schedule_params
        )        
        self.beta = torch.tensor(beta, device=device, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.bar_alpha = torch.cumprod(self.alpha.clone(), 0)

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

    # -------------------------------------------------------------------
    # Training

    def add_noise(self, x0, t=None, eps=None):
        """
        Adds noise to a clean sample to assist in DDPM training.

        We don't ask you to implement the scheduler, because it's not particularly interesting
        and is a little fiddely. The gist is a given noise schedule, alpha tells you how much
        variance a sample at an arbitrary point in time along the diffusion process will have.
        That lets you sample from the distribution at that point directly, instead of needing to
        forward-propogate the diffusion process many times during training.

        You need to implement the part that calculates the noisy sample.

        This method should return :
        - xt (noisy sample at t)
        - t 
        - eps (the added Gaussian noise at diffusion step t).

        Hint 1 - eps is not dependent on t. Why? eps ~ N(0, I)。Adding noise is handled by $\sqrt{1-\bar\alpha_t}$, indep of t
        Hint 2 - see page 3 and 4 of "Denoising Diffusion Probabilistic Models" Ho et al. 2020.

        """
        # Randomly samples a point along the reverse-diffusion process t
        # In the paper, t ∼ Uniform({1, . . . , T}) 
        # Don't use torch.uniform -> Cause this is continous
        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device) if t is None else t
        # Samples the alpha (time-dependent variance set by the scheduler)
        bar_alpha = at_least_ndim(self.bar_alpha[t], x0.dim())

        # TODO: fill in remaining code
        
        # x0: (batch, horizon, action_dim)
        # t: (batch, )
        # eps: (batch, horizon, action_dim)
        # For b-th sample, h-th future action, a-th action scale
        
        
        # paper: eps ∼ N (0, I)
        if eps is None:
            eps = torch.randn_like(x0) # device is will auto be the same as x
        
        # Reparameterization from formula 4
        # we get # \[x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\]
        xt = (
            bar_alpha.sqrt() * x0
            + (1.0 - bar_alpha).sqrt() * eps
        )
        return xt, t, eps

    def loss(self, x0, condition=None):
        """
        Defines the loss function that will be used by loss.backwards(). 
        
        x0 is the original input sample, shaped (batch_size,action_chunk_length,action_dim)

        Note - this should be able to handle both epsilon (noise) prediction and x (clean) prediction.
        You can find which mode the current DDPM agent is in by checking self.predict_noise.
        """

        # generates the condition vector (in our case, the embeddings from video)
        condition_vec = self.model["condition"](condition) if condition is not None else None
        
        # My Understanding:
        # We take a clean action x0, pick a random timestep t, and mix in some random Gaussian noise eps 
        # using the formula xt = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·eps. The bigger t is, the noisier the result. 
        # Then we show this noisy sample to the network and ask it, “what noise did I just add?” 
        # we compare its guess to the real eps with an MSE loss and update the network, 
        # so over time it gets really good at spotting noise. 
        # Once it’s trained, generating something is in reverse: start from pure random noise, 
        # ask the network to predict the noise, remove a little of it, and repeat this step by step.
        # TODO fill in 
        xt, t, eps = self.add_noise(x0)
        
        pred = self.model["diffusion"](
            xt, t, condition_vec
        )
        
        target = eps if self.predict_noise else x0
        
        return F.mse_loss(pred, target)

    def update(self, x0, condition=None, update_ema=True):
        loss = self.loss(x0, condition)
        loss.backward()
        grad_norm = (
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) if self.grad_clip_norm else None
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema:
            self.ema_update()
        return {"loss": loss.item(), "grad_norm": grad_norm}

    # -------------------------------------------------------------------
    # Sampling

    @torch.no_grad()
    def sample(self, 
               prior: torch.Tensor, 
               n_samples: int = 1, 
               sample_steps: int = None,
               use_ema: bool = True, 
               temperature: float = 1.0, 
               condition=None,
               record_sampling = False):
        """Ancestral DDPM reverse-diffusion sampler. `prior` sets the sample
        shape (n_samples, horizon, action_dim); its values are only used
        where the (unused, in this repo) fix_mask would apply, so here it
        just fixes the shape/device/dtype."""
        if sample_steps is not None and sample_steps != self.diffusion_steps:
            warnings.warn("sample_steps != diffusion_steps; DDPM's ancestral sampler always runs the full loop.")
        model = self.model_ema if use_ema else self.model

        condition_vec = model["condition"](condition) if condition is not None else None

        xt = torch.randn_like(prior, device=self.device) * temperature

        if record_sampling:
            sample_record = []

        for t in range(self.diffusion_steps - 1, -1, -1):
            if record_sampling:
                sample_record.append(xt)
            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[t]
            bar_alpha_prev = self.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            alpha = self.alpha[t]
            beta = self.beta[t]

            pred = model["diffusion"](xt, t_batch, condition_vec)
            
            # TODO Fill in next bits of code to implement the reverse-diffusion process.
            # Note this needs to be able to handle both epsilon and x prediction.
            if self.predict_noise:
                # by the paper formula 4: $$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$
                # we can rearragne it to get: x_0 = .... as the following:
                x0_pred = (xt - (1.0 - bar_alpha).sqrt() * pred) / bar_alpha.sqrt()
            else:
                x0_pred = pred
            
            if self.clip_action:
                x0_pred = x0_pred.clamp(-1.0, 1.0)
            
            # paper: formula 7
            mean = (
                bar_alpha_prev.sqrt()
                * beta
                / (1.0 - bar_alpha)
                * x0_pred
                +
                alpha.sqrt()
                * (1.0 - bar_alpha_prev)
                / (1.0 - bar_alpha)
                * xt
            )
            
            # paper: 3.2 Reverse process 
            variance = (
                beta
                * (1.0 - bar_alpha_prev)
                / (1.0 - bar_alpha)
            )
            
            # paper: Algo2.
            if t > 0:
                xt = (
                    mean
                    + variance.clamp_min(0.0).sqrt()
                    * torch.randn_like(xt)
                    * temperature
                )
            else:
                xt = mean
                
        if record_sampling:
            sample_record.append(xt)
            return xt, sample_record
        
        return xt
