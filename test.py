import mujoco_py

model = mujoco_py.load_model_from_path("RobotEnv/assets/UR5/robotModel.xml")

sim = mujoco_py.MjSim(model)

viewer = mujoco_py.MjViewer(sim)

t = 0

while True: 

    t += 1
    sim.step()
    viewer.render()

    if t > 500:
        break


