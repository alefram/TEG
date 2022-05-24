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
target_bounds = np.array(((-0.3, 0.1), (-0.3, 0.3), (0.45, 0.5)), dtype=object)
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

    for i in range(episodes):
        print('---------------------')
        print("episodio", i)
        print('---------------------')

        goal = simulation.random_target(target_bounds, geom_pos, sim)
        controller.reset()

        controller.move_to(np.array(goal), distance_threshold=dist, timer=timer)

if __name__ == "__main__":
    main()


#
# env = UR5_EnvTest(simulation_frames=1, torque_control= 0.01, distance_threshold=0.05, gui=True)
# ac =  torch.load("/home/alexis/Documentos/repos/TEG/agents_old/ddpg4/pyt_save/model.pt")

# for i_episode in range(50):
#     print('---------------------------------------------------------------')
#     print("estoy en pisodio",i_episode)
#     print('---------------------------------------------------------------')
#
#     observation = env.reset()
#
#     for t in range(500):
#
#         env.render()
#
#         # action = env.action_space.sample() #agente random
#         action = ac.act(torch.as_tensor(observation, dtype=torch.float16)) # agente vpg
#
#         observation, reward, done, info = env.step(action)
#
#         if done:
#             print("la tarea es resuelta en:", t * 0.002)
#             break
#
# env.close()
