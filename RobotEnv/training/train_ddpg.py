import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest #acuerdate de agregar TEG para colab
from spinup import ddpg_pytorch as ddpg
import spinup.algos.pytorch.ddpg.core as core
import torch.nn as nn
import time

#agente thursday
# env = lambda: UR5_EnvTest(simulation_frames=4, torque_control= 0.01, distance_threshold=0.05, gui=False)

# logger_kwargs = dict(output_dir='agents/agents1', exp_name='politica 1')
# ac_kwargs = dict(hidden_sizes=[32,32], activation=nn.ReLU)


# ddpg(env, actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs, seed=0,
#     steps_per_epoch=4000, epochs=150, replay_size=1000000, gamma=0.99,
#     polyak=0.995, pi_lr=0.001, q_lr=0.001, batch_size=100, start_steps=10000,
#     update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
#     max_ep_len=500, logger_kwargs=logger_kwargs, save_freq=1)


#agente friday
# env = lambda: UR5_EnvTest(simulation_frames=4, torque_control= 0.01, distance_threshold=0.05, gui=False)

# logger_kwargs = dict(output_dir='agents/friday', exp_name='politica 2')
# ac_kwargs = dict(hidden_sizes=[100,100], activation=nn.ReLU)


# ddpg(env, actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs, seed=0,
#     steps_per_epoch=4000, epochs=100, replay_size=1000000, gamma=0.99,
#     polyak=0.995, pi_lr=0.001, q_lr=0.001, batch_size=100, start_steps=10000,
#     update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
#     max_ep_len=500, logger_kwargs=logger_kwargs, save_freq=1)

#agente sunday
# env = lambda: UR5_EnvTest(simulation_frames=4, torque_control= 0.01, distance_threshold=0.05, gui=False)

# logger_kwargs = dict(output_dir='agents/agent1', exp_name='politica 1')
# ac_kwargs = dict(hidden_sizes=[400,300], activation=nn.ReLU)


# #inicio
# start = time.time()

# ddpg(env, actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs, seed=0,
#     steps_per_epoch=4000, epochs=120, replay_size=1000000, gamma=0.99,
#     polyak=0.995, pi_lr=0.001, q_lr=0.001, batch_size=100, start_steps=10000,
#     update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
#     max_ep_len=500, logger_kwargs=logger_kwargs, save_freq=1)

# #final
# end = time.time()

# execution_time = (end - start)/60

# print("Tiempo de execución:", execution_time, "minutes")

if __name__ == '__main__':
    # entrenar
    env = lambda: UR5_EnvTest(simulation_frames=4, torque_control= 0.01, distance_threshold=0.05, gui=False)

    logger_kwargs = dict(output_dir='agents/ddpg4', exp_name='robot_train2')
    ac_kwargs = dict(hidden_sizes=[800,500], activation=nn.ReLU)


    ddpg(env, actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs, seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=1000000, gamma=0.99,
        polyak=0.995, pi_lr=0.001, q_lr=0.001, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
        max_ep_len=500, logger_kwargs=logger_kwargs, save_freq=1)
