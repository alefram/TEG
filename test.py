import mujoco_py
import argparse

parser = argparse.ArgumentParser(description="UR5")
parser.add_argument(dest='modelo', type=str, default="robotModel.xml", help="robot a utilizar")
args = parser.parse_args()

model = mujoco_py.load_model_from_path("RobotEnv/assets/UR5/" + args.modelo)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

t = 0

while True:

    t += 1
    sim.step()
    print(sim.data.qpos)

    viewer.render()

    if t > 500:
        break
