from TEG.envs.UR5_random_target_v0 import UR5_EnvTest
import TEG.tools.simulation as sim

env = UR5_EnvTest(simulation_frames=5, torque_control= 0.01, distance_threshold=0.05)

def main():
    for episode in range(5):
        print("episode {}".format(episode))
        env.reset()

        for t in range(1000):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    return env.robot, env.sim

if __name__ == '__main__':
    sim.render_simulation(loader=main)

