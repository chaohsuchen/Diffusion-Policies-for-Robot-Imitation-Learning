import os
import shutil
import subprocess
import sys

if "MUJOCO_GL" not in os.environ and sys.platform != "darwin":
    # Pick a headless-friendly OpenGL backend for mujoco's offscreen renderer:
    # EGL on machines with an NVIDIA GPU, OSMesa (software) otherwise. Both
    # work without a GLX-capable display, unlike the default GLFW backend.
    def _has_nvidia_gpu():
        return shutil.which("nvidia-smi") is not None and subprocess.run(
            ["nvidia-smi"], capture_output=True
        ).returncode == 0

    os.environ["MUJOCO_GL"] = "egl" if _has_nvidia_gpu() else "osmesa"

from gym.envs.registration import register
register(
    id='ballcatch-v0',
    entry_point='gym_thing.envs:ThingBallCatchEnv',
    max_episode_steps=1000,
)

register(
    id='thing-v0',
    entry_point='gym_thing.envs:ThingEnv',
)

