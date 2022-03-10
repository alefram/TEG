import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest
import mujoco_py


env = UR5_EnvTest(simulation_frames=10, torque_control= 0.01, distance_threshold=0.5, Gui=True)

for i_episode in range(20):
    print("estoy en pisodio",i_episode)
    env.reset()


    for t in range(500):
        print("paso ", t)

        env.render()

        action = env.action_space.sample()
        print("Action del agente",action)
        observation, reward, done, info = env.step(action)
        print('-----------------')
        print('observacion')
        print('posicion de la garra:',info['gripper_position'])
        print('posicion de las articulaciones:', info['j_position'])
        print('velocidad de las articulaciones:', info['j_velocity'])
        print('distancia a la meta:', info['dist'])
        print('recompens:', reward)
        print('---------------------------------')

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
