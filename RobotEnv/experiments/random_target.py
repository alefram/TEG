import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest
import mujoco_py
import torch

env = UR5_EnvTest(simulation_frames=1, torque_control= 0.01, distance_threshold=0.05, gui=True)
ac =  torch.load("/home/alexis/Documentos/repos/TEG/agents_old/ddpg4/pyt_save/model.pt")

for i_episode in range(50):
    print('---------------------------------------------------------------')
    print("estoy en pisodio",i_episode)
    print('---------------------------------------------------------------')

    observation = env.reset()

    for t in range(500):

        env.render()

        # action = env.action_space.sample() #agente random
        action = ac.act(torch.as_tensor(observation, dtype=torch.float16)) # agente vpg

        observation, reward, done, info = env.step(action)
        
        if done:
            print("la tarea es resuelta en:", t * 0.002)
            break

env.close()
