from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg_pytorch as ddpg
from RobotEnv.envs.UR5_Env import UR5_EnvTest
import spinup.algos.pytorch.ddpg.core as core
import torch.nn as nn
import gym
import time

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu',type=int, default=3)
    parser.add_argument('--num_runs', type=int, default=1) 
    args = parser.parse_args()

    # ambiente
    env = lambda: UR5_EnvTest(simulation_frames=4, 
                            torque_control= 0.01, 
                            distance_threshold=0.05, 
                            gui=False)

    #config
    eg = ExperimentGrid(name='agentes')
    eg.add('env_fn', env, '', False)
    eg.add('actor_critic', core.MLPActorCritic)
    eg.add('ac_kwargs:hidden_sizes', 
        [(400,300), (500,400), (600,500)], 
        'hid'
    )
    eg.add('ac_kwargs:activation', nn.ReLU)
    eg.add('logger_kwargs:output_dir', 
        ['agents/agent3', 'agents/agent4', 'agents/agent5']
    )
    eg.add('logger_kwargs:exp_name',['agente3','agente4','agente5'])
    eg.add('seed', 0)
    eg.add('epochs', 100)
    eg.add('steps_per_epoch', 4000)
    eg.add('replay_size', 1000000)
    eg.add('gamma', 0.99) #factor de descuento
    eg.add('polyak', 0.995) 
    eg.add('pi_lr', 0.001)
    eg.add('q_lr', 0.001)
    eg.add('batch_size', 128)
    eg.add('start_steps', 10000)
    eg.add('update_after', 1000)
    eg.add('update_every', 50)
    eg.add('act_noise', 0.1)
    eg.add('num_test_episodes', 5)
    eg.add('max_ep_len', 500)
    eg.add('save_freq', 1)

    #ejecutar entrenamiento
    eg.run(ddpg, num_cpu=args.cpu)
