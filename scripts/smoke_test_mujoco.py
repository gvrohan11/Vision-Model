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