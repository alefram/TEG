# TEG ![Status badge](https://img.shields.io/badge/status-in%20progress-important)

This is my Bachelor Thesis, where is a set of environments and tools to develop
smart controllers for robot manipulators using Reinforcement Learning.

### 🚀 Installation

This project use python 3.7+ and it requires mujoco installation you can check
[HERE](https://github.com/deepmind/mujoco).

>**NOTE**
>Currently, this project is using mujoco-py but I hope soon will change it to the mujoco
>native python binding from Deepmind. 

```bash

git clone https://github.com/Alexfm101/TEG.git 
cd TEG
python -m pip install -e .

```

### Example

```python

from RobotEnv.envs.UR5_Env import UR5_EnvTest

env = UR5_EnvTest(simulation_frames=5, torque_control= 0.01, distance_threshold=0.05, gui=True)

for episode in range(20):
    env.reset()

    for t in range(500):
        
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()


```

### TODO



### 🧾 License

The Apache 2.0 License
 
    
