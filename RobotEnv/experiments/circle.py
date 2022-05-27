"""experimento 1 """

# import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest
# import mujoco_py
# import torch
from RobotEnv.tools import simulation
from RobotEnv.tools import controllers
from RobotEnv.tools.logger import Logger
import numpy as np
import os
import argparse

import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--agent", help="selecionar agente")
parser.add_argument("--dist", \
                    help="distancia minima para lograr la tarea", \
                    type=float)
parser.add_argument("--render", help="mostrar simulación")
parser.add_argument("-t", "--timer", \
                    help="tiempo de duración del controlador ajustando", \
                    type=int)
parser.add_argument("-i", "--episodes", help="episodios", type=int)

args = parser.parse_args()

# parametros
dist = args.dist
render = args.render
geom_pos = 1
timer = args.timer
episodes = args.episodes
agent = args.agent

# Simulador
sim = simulation.create_simulation("robotModelV2.xml")

# controlador
controller = controllers.Manipulator_Agent(agent, sim, render=render)

# recolector de data
logger = Logger()

def main():

    #circulo
    theta = np.linspace(0, 2*np.pi, 50)
    r = np.sqrt(0.04)
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    #datos
    posx = []
    posy = []
    posz = []
    datax = []
    datay = []

    for i in range(episodes):
        print('---------------------')
        print("episodio", i)
        print('---------------------')

        controller.reset()

        for i in range(len(theta)):

            target = np.array([x[i]-0.1, y[i], 0.5])
            simulation.post_target(sim, target, geom_pos)

            position, qpos, control, _, _ = controller.move_to(np.array(target),
                                distance_threshold=dist,
                                timer=timer
                                )
            obs = controller.observe()
            datax.append(obs[0])
            datay.append(obs[1])

            posx.extend(position["pos_x"])
            posy.extend(position["pos_y"])
            posz.extend(position["pos_z"])

    # grafica
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, 0.5, 'o', linestyle='-', label='objetivo')
#    ax.plot(posx, posy, posz, label='trayectoria total')
    ax.plot(datax, datay, 0.5, 'v', linestyle='--' ,label="trayectoria")
    ax.legend()

    

    plt.show()


if __name__ == "__main__":
    main()
