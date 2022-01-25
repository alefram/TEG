import gym
from RobotEnv.envs.UR5_Env import UR5_EnvTest
import mujoco_py

env = UR5_EnvTest(400,400,0.2,100,100, True)
# viewer = mujoco_py.MjViewer(env.sim)



for i_episode in range(20):
    print("estoy en pisodio")
    observation = env.reset()
    for t in range(100):
        print("estoy en t")
        
        # if env.render:
        #     viewer.render()

        env.render()
            
        print(observation)
        action = env.action_space
        observation, reward, done = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
