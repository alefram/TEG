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

parser = argparse.ArgumentParser()

parser.add_argument("--agent", help="selecionar agente")
parser.add_argument("--dist", help="distancia minima para lograr la tarea", type=float)
parser.add_argument("--render", help="mostrar simulación")
parser.add_argument("-t", "--timer", help="tiempo de duración del controlador ajustando", type=int)
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

    # recta
    x =  [i/10 for i in range(-2,2)]
    y = x
    for i in range(episodes):
        print('---------------------')
        print("episodio", i)
        print('---------------------')

        controller.reset()

        for i in range(len(x)):

            target = np.array([x[i], y[i], 0.5])
            simulation.post_target(sim, target, geom_pos)

            controller.move_to(np.array(target), distance_threshold=dist, timer=timer)


if __name__ == "__main__":
    main()
