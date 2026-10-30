# Attribution and project scope

This repository grew out of **CSC2626: Imitation Learning for Robotics** coursework. The original assignment handout, private submission archive, and local homework guide are not part of the public project.

## Implementation boundary

The course scaffold supplied neural-network backbones, observation/data utilities, and training/evaluation integration. The course implementation work includes forward DDPM noising, epsilon and clean-sample losses, reverse sampling, and action chunk execution. The repository also contains an explicit single-action BC baseline, execution/checkpoint checks, platform-aware device selection, and Slurm scripts. The portfolio presentation adds a reproducible analysis of archived evaluations and documents the sigmoid noise-schedule experiment. This description does not claim sole authorship of all bundled source files.

## Methods and software

- **Diffusion Policy**: Cheng Chi and collaborators, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. Project: https://diffusion-policy.cs.columbia.edu/
- **DDPM**: Jonathan Ho, Ajay Jain, Pieter Abbeel, *Denoising Diffusion Probabilistic Models* (2020). https://arxiv.org/abs/2006.11239
- **Robomimic**: human demonstration datasets and environment interfaces. https://github.com/ARISE-Initiative/robomimic
- **Robosuite / MuJoCo**: robot manipulation simulation. https://github.com/ARISE-Initiative/robosuite and https://github.com/google-deepmind/mujoco
- **CleanDiffuser**: the supplied flat diffusion-policy pipeline identifies CleanDiffuser's Robomimic pipeline as its basis. https://github.com/CleanDiffuserTeam/CleanDiffuser

Dataset and model-framework rights remain with their respective owners. No new blanket license is asserted over course-provided or third-party material. The dataset is downloaded separately and is not redistributed here.

The README's `diffusion_text.gif` links to the author's companion [DDPM From Scratch](https://github.com/chaohsuchen/DDPM-From-Scratch) project. It illustrates that separate project, not a robot task rollout.
