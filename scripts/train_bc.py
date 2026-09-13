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
# a training policy here is a nn: 
# input: robots observations (OBS_KEYS) 
# output: the robot's actions (7 numbers to control the robot)

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
# We will use MSE and Adam optimizer to train the policy
# guess -> compare -> backprop -> update weights -> repeat
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
N, batch = X.shape[0], 256

for epoch in range(10):
    perm = torch.randperm(N)
    running_loss = 0.0
    for i in range(0, N, batch):
        idx = perm[i : i + batch] # randomly select a batch of indices
        pred = policy(X[idx]) # predict the robot's actions given the robot's observations
        loss = loss_fn(pred, Y[idx]) # compared predicted vs expected
        optimizer.zero_grad() # clear grads from last step
        loss.backward() # compute gradients for each weight via backprop
        optimizer.step() # update weights using grads and lr
        running_loss += loss.item() * len(idx) # accumulate loss for this batch
    print(f"Epoch {epoch}: loss = {running_loss/N:.4f}")
