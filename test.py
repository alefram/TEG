import mujoco_py
import argparse

#indicaciones para ingresar por el usuario
parser = argparse.ArgumentParser(description="UR5")
parser.add_argument(dest='modelo', type=str, default="robotModel.xml", help="robot a utilizar")
args = parser.parse_args()

#creacion del modelo del robot
model = mujoco_py.load_model_from_path("RobotEnv/assets/UR5/" + args.modelo)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

t = 0 #paso de tiempo

sim_state = sim.get_state()

while True:

    t += 1
    sim.set_state(sim_state)

    for  i in  range(1000):
        if i < 150:
            sim.data.ctrl[:] = 0.0
        else:
            sim.data.ctrl[:] = 1.0


        sim.step()
        print(sim.data.qpos)
        viewer.render()

    if t > 500:
        break
