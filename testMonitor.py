import gym
from gym.wrappers import Monitor
from RobotEnv.envs.UR5_Env import UR5_EnvTest

env = Monitor(UR5_EnvTest(simulation_frames=10,Gui=True), './video', force=True)
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    env.render()
    state_next, reward, done, info = env.step(action)
env.close()

