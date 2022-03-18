import mujoco_py
import argparse
import numpy as np
import torch
#indicaciones para ingresar por el usuario
parser = argparse.ArgumentParser(description="UR5")
parser.add_argument(dest='modelo', type=str, default="robotModel.xml", help="robot a utilizar")
args = parser.parse_args()

#creacion del modelo del robot
model = mujoco_py.load_model_from_path("RobotEnv/assets/UR5/" + args.modelo)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

t = 0 #paso de tiempo

ac =  torch.load("/home/alexis/Documentos/repos/TEG/agents/ddpg1/pyt_save/model.pt")


sim_state = sim.get_state()

while True:

    t += 1
    sim.set_state(sim_state)


    for  i in  range(100000):
        if i < 150:
            sim.data.ctrl[:] = 0.0
        else:
            # action = ac.act(torch.as_tensor(sim_state, dtype=torch.float3)) # agente vpg

            sim.data.ctrl[:] = 1


        sim.step()
        print(sim.data.qpos)
        viewer.render()

    if t > 500:
        break
