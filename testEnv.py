import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest
import mujoco_py

env = UR5_EnvTest(100,50,True)


for i_episode in range(20):
    print("estoy en pisodio",i_episode)
    env.reset()

    for t in range(100):
        print("paso ", t)

        env.render()

        action = env.action_space.sample()
        print("Action del agente",action)
        observation, reward, done = env.step(action)
        print("observación", observation)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
