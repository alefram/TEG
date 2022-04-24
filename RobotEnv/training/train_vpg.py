import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest #acuerdate de agregar TEG para colab
from spinup import vpg_pytorch as vpg
import spinup.algos.pytorch.vpg.core as core


if __name__ == '__main__':
    # entrenar
    env = lambda: UR5_EnvTest(simulation_frames=10, torque_control= 0.01, distance_threshold=0.5, Gui=False)

    logger_kwargs = dict(output_dir='agents/vpg/', exp_name='robot_train2')

    vpg(env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=1000, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80,lam=0.97, max_ep_len=1000,
        logger_kwargs=logger_kwargs, save_freq=10)
