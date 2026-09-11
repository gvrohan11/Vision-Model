import h5py
import numpy as np

# Download 200 demos and analyzed them
# We observed what the robot saw and what the robot did at each step
# (_,_) = (observation, action)
# (# of timesteps in this demo, how many things the robot needs to observe or do at each step)


DATASET = "data/lift/ph/low_dim_v141.hdf5"
f = h5py.File(DATASET, "r")

demos = list(f["data"].keys())
print(f"Number of demos: {len(demos)}")
print(f"Total recorded timesteps across all demos: {f['data'].attrs['total']}")

demo = f['data']['demo_0'] # get the first demo
print()
print("------DEMO 0------")
print(f"Length (timesteps in this demo): {demo.attrs['num_samples']}")

actions = demo['actions']
print(f"Shape of actions: {actions.shape}") # (timesteps, 7 numbers per step)
print(f"Expert's action at the first timestep: {np.round(actions[0], 3)}")

print()
print("What the robot saw at each step (observation keys):")
for key in demo['obs'].keys():
    print(f"  {key} -> {demo['obs'][key].shape}")
f.close()

# key = "robot0_joint_pos" -> the robot's joint positions (7 numbers per step)
# the 2nd integer in the (,) pair represents the 