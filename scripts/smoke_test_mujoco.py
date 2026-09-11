
####################################### THE PHYSICS WORLD #######################################
# this is a little simulation / world where gravity, weight, and collisions behave like real life


import mujoco

# tiny hand-written world (one floor, one box)
# mujoco describes worlds in xml format called mjcf (mujoco xml format)

MODEL_XML = """
    <mujoco>
        <worldbody>
            <light pos="0 0 3"/>
                <geom name="floor" type="plane" size="2 2 0.1"/>
                <body name="box" pos="0 0 1">
                    <joint name="free" type="free"/>
                    <geom name="box" type="box" size="0.1 0.1 0.1"/>
                </body>
        </worldbody>
    </mujoco>
"""

model = mujoco.MjModel.from_xml_string(MODEL_XML) # fixed world
data = mujoco.MjData(model) # live state (positions, velocities, etc) that changes over time

print(f"Height at the start: {round(data.qpos[2], 3)}") # qpos[2] is the box's height

for i in range(500):
    # in each mj_step, gravity pulls the box down, and the box will eventually hit the floor and stop falling
    mujoco.mj_step(model, data) # step the simulation forward 
    print(f"Height at step {i}: {round(data.qpos[2], 3)}") # print the box's height (the height becomes 0.1 at step 286)

####################################### A Robot Arm in the Physics World #######################################
# This is a simulated arm sitting in a room with a cube, and a task defined ("lift the cube")

import numpy as np
import robosuite as suite

env = suite.make(
    env_name="Lift", # task name: lift cube off floor
    robots="Panda", # the arm doing it
    has_renderer=False, # no on-screen window
    has_offscreen_renderer=False, # we still render offscreen, for the camera observations
    use_camera_obs=False, # give us numbers, not images, for now
    control_freq=20 # how many times per second we control robot and step simulation
)

obs = env.reset() # reset env, get first observation
print(f"The number of things the robot observes: {len(obs)}") # obs is a dict, with keys for each type of observation
print(f"Some of the sensor names: {list(obs.keys())[:5]}") # print the first 5 keys in the observation dict
print(f"robot arm joint positions: {np.round(obs['robot0_joint_pos'], 3)}") # print the robot's joint positions

action_dim = env.action_dim # how many numbers we need to give the robot to control it
print(f"How many numbers control the arm each step: {action_dim}")

for i in range(5):
    action = np.random.uniform(low=-1, high=1, size=action_dim) # random action
    obs, reward, done, info = env.step(action) # take a step in the environment
    print(f"Step: {i}, reward: {round(reward, 4)}")

env.close()
print("Done with smoke test!")

'''
Smoke test result:
The robot observes 15 things (joint positions, velocities, end-effector position, gripper state, cube position, etc.)
7 joint positions
7 control numbers per step (how many #s a policy has to output each step to control the robot)
Currently reward 0.0 each step
'''