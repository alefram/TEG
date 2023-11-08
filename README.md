# TEG 

TEG is a straightforward environment for Reinforcement Learning that enables 
the training of RL agents for a robot manipulator. It's based on the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
and [Mujoco](https://github.com/deepmind/mujoco).

## Installation

This project use python 3.7+

You can install it by using pip

```bash
pip install TEG
```

Or manually cloning the github repository

```bash

git clone https://github.com/Alexfm101/TEG.git 
cd TEG
python -m pip install -e .

```

## Example

TEG environment are simple Python `env` classes to allow an AI agent to interact
with them very simple. Here's an example:

```python
from TEG.envs.UR5_v0 import UR5Env_v0

env = UR5Env_v0(simulation_frames=5, torque_control= 0.01, distance_threshold=0.05)

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
 
