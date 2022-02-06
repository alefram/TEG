import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest
import mujoco_py

env = UR5_EnvTest(400,400,100,100000000,100, True)



for i_episode in range(20):
    print("estoy en pisodio",i_episode)
    observation = env.reset()
    for t in range(100):
        print("paso ", t)

        env.render()

        print("observación", observation)
        action = env.action_space
        observation, reward, done = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
