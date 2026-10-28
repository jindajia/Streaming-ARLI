"""gym_aloha 0.1.1 AlohaTransferCube-v0 (as registered in dsrl_pi0 train_sim.py), with explicit seeding.

* control timestep 0.02 s (50 Hz), episode length 400 steps, success = reward 4 (left gripper holds the cube
  off the table); gym_aloha terminates the episode on success.
* Box pose is a deterministic function of ``reset(seed)``.
* Optional speed-only patch ``render_only_top`` (skip the two unused cameras) is OFF: in scripts/check_r1.py the
  ``top`` image was not bit-identical with it (gain only 2.6 -> 2.1 ms/step), so the unmodified simulator is used.
"""

from __future__ import annotations

import collections

import numpy as np
import PIL.Image

MAX_STEPS = 400
MAX_REWARD = 4


def _patch_render_only_top():
    from gym_aloha.tasks import sim

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {"top": physics.render(height=480, width=640, camera_id="top")}
        return obs

    sim.BimanualViperXTask.get_observation = get_observation


def make_env(render_only_top: bool = False):
    if render_only_top:
        _patch_render_only_top()
    from gym_aloha.env import AlohaEnv
    return AlohaEnv(task="transfer_cube", obs_type="pixels_agent_pos", render_mode="rgb_array")


def rl_image(obs, size: int = 64) -> np.ndarray:
    """cam_high (= gym_aloha 'top') resized to size x size, as dsrl_pi0 obs_to_img (PIL bilinear-free resize)."""
    return np.array(PIL.Image.fromarray(obs["pixels"]["top"]).resize((size, size)))


def qpos(obs) -> np.ndarray:
    return np.asarray(obs["agent_pos"], dtype=np.float32)
