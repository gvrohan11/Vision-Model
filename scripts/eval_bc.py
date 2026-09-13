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

policy = nn.Sequential(
    nn.Linear(19, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 7)
)

policy.load_state_dict(torch.load("bc_policy.pt"))
policy.eval()

N_EPISODES = 50
HORIZON = 400
successes = 0
for ep in range(N_EPISODES):
    obs = env.reset()
    lifted = False
    for t in range(HORIZON):
        vec = obs_to_vec(obs)
        with torch.no_grad():
            action = policy(torch.tensor(vec)).numpy()
        obs, reward, done, info = env.step(action)
        if env._check_success():
            lifted = True
            break
    successes += int(lifted)
    if lifted:
        print(f"Episode {ep:2d}: Success")
    else:
        print(f"Episode {ep:2d}: Fail")
success_rate = 100 * (successes / N_EPISODES)
print(f"Success rate: {successes}/{N_EPISODES} = {success_rate:.0f}%")
