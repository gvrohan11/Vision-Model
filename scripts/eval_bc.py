import numpy as np
import torch
import torch.nn as nn
import robosuite as suite

env = suite.make(
    env_name="Lift", # task name: lift cube off floor
    robots="Panda", # the arm doing it
    has_renderer=False, # no on-screen window
    has_offscreen_renderer=False, # we still render offscreen, for the camera observations
    use_camera_obs=False, # give us numbers, not images, for now
    use_object_obs=True, # give us numbers for cube's position and orientation
    control_freq=20 # how many times per second we control robot and step simulation
)

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

def obs_to_vec(obs):
    return np.concatenate([obs[k] for k in OBS_KEYS]).astype("np.float32") # 19 minutes