# TEG ![Status badge](https://img.shields.io/badge/status-in%20progress-important) 
 <!--[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)-->

TEG, is a environment to develop AI agents for robot 
manipulators using Reinforcement Learning. It's based on the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
and [Mujoco](https://github.com/deepmind/mujoco) libraries.

## Installation

This project use python 3.7+
```bash

git clone https://github.com/Alexfm101/TEG.git 
cd TEG
python -m pip install -e .

```

## Example

TEG environments are simple Python `env` classes to allow an AI agent to interact
with them very simple. Here's an example:

```python
from TEG.envs.UR5.RandomTrayectoryV0 import UR5_EnvTest

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
    main()
```

## 🧾 License

The Apache 2.0 License
 
