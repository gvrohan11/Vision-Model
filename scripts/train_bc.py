import h5py
import numpy as np
import torch
import torch.nn as nn

# FIRST: Load the demos into the 2 big arrays and pick 4 observation keys as the policy's input

DATASET = "data/lift/ph/low_dim_v141.hdf5"
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

f = h5py.File(DATASET, "r")
obs_list, act_list = [], []
for demo in f['data'].keys():
    g = f['data'][demo]
    obs = np.concatenate([g['obs'][k][:] for k in OBS_KEYS], axis=1)
    obs_list.append(obs)
    act_list.append(g['actions'][:])
f.close()

X = torch.tensor(np.concatenate(obs_list, axis=0), dtype=torch.float32)
Y = torch.tensor(np.concatenate(act_list, axis=0), dtype=torch.float32)
print(f"Observations: {X.shape}, Actions: {Y.shape}")

# SECOND: Define a simple nn policy
# nn.Sequential runs the models in order
# first layer: input -> 256 hidden units, ReLU activation
# second layer: 256 hidden units -> 256 hidden units, ReLU activation
# third layer: 256 hidden units -> output (7 numbers to control the robot)

policy = nn.Sequential(
    nn.Linear(X.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, Y.shape[1])
)

# THIRD: Training loop
