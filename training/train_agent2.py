"""Entrenamiento del agente 2"""

import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest #acuerdate de agregar TEG para colab
from spinup import ddpg_pytorch as ddpg
import spinup.algos.pytorch.ddpg.core as core
import torch.nn as nn
import time


if __name__ == '__main__':
    # ambiente
    env = lambda: UR5_EnvTest(simulation_frames=4, torque_control= 0.01, distance_threshold=0.05, gui=False)
    
    # agregar parametros de la red neuronal y donde guardar.
    logger_kwargs = dict(output_dir='../agents/_Agent2_', exp_name='_Agent2_')
    ac_kwargs = dict(hidden_sizes=[600,500], activation=nn.ReLU)

    # tiempo inicio
    start = time.time()

    # algoritmo de entrenamie
    ddpg(env, actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs, seed=0,
        steps_per_epoch=4000, epochs=150, replay_size=1000000, gamma=0.99,
        polyak=0.995, pi_lr=0.001, q_lr=0.001, batch_size=128, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=5,
        max_ep_len=500, logger_kwargs=logger_kwargs, save_freq=1)

    # tiempo final
    end = time.time()

    execution_time = (end - start)/60
    print("Tiempo de execución:", execution_time, "minutes")

    # duro 1099.569 minutos