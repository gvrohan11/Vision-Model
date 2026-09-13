import numpy as np
import imageio
import os
import torch
import torch.nn as nn
import robosuite as suite
from robosuite.controllers import load_controller_config

VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# this script evaluates the policy trained in train_bc.py
# the policy is a nn that takes in a robots observations and outputs the robots actions
# the eval loop runs the policy 50 times and counts how many times the arm lifts the cube off the floor

controller_config = load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    env_name="Lift", # task name: lift cube off floor
    robots="Panda", # the arm doing it
    controller_configs=controller_config, # how we control the bot
    has_renderer=False, # no on-screen window
    has_offscreen_renderer=True, # we still render offscreen, for the camera observations
    use_camera_obs=True, # give us numbers, not images, for now
    use_object_obs=True, # give us numbers for cube's position and orientation
    camera_names="agentview", # a fixed 3rd-person camera
    camera_heights=256,
    camera_widths=256,
    control_freq=20 # how many times per second we control robot and step simulation
)

EVAL_OBS_KEYS = ["object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

def obs_to_vec(obs):
    return np.concatenate([obs[k] for k in EVAL_OBS_KEYS]).astype("float32") # 19 minutes

policy = nn.Sequential(
    nn.Linear(19, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 7)
)

policy.load_state_dict(torch.load("bc_policy.pt"))
policy.eval()

N_EPISODES = 5
HORIZON = 400
successes = 0
for ep in range(N_EPISODES):
    obs = env.reset()
    frames = []
    lifted = False
    for t in range(HORIZON):
        vec = obs_to_vec(obs)
        with torch.no_grad():
            action = policy(torch.tensor(vec)).numpy()
        obs, reward, done, info = env.step(action)
        frames.append(obs["agentview_image"][::-1])
        if env._check_success():
            lifted = True
            break
    successes += int(lifted)
    if lifted:
        tag = "Success"
    else:
        tag = "Fail"
    imageio.mimsave(os.path.join(VIDEO_DIR, f"rollout_ep{ep}_{tag}.mp4"), frames, fps=20)
    print(f"Episode {ep}: {tag.upper()} {len(frames)} frames")

    
print("Videos saved in project folder")
