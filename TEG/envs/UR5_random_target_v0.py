import mujoco
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces

# Convert the observation of the environment into the observation space with its limits
def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float16)
        high = np.full(observation.shape, float("inf"), dtype=np.float16)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space



class UR5_EnvTest(gym.Env):
    """
    ### initial arguments:

    1. simulation_frames: number of simulation steps using one action of the agent
    2. torque_control: constant used in the reward system to control torque
    3. distance_threshold: final distance which indicates that the claw has reached the target.
    """
    def __init__(self,
                simulation_frames,
                torque_control,
                distance_threshold):

        #Init configurations
        self.simulation_frames = simulation_frames
        self.C_a = torque_control
        self.distance_threshold = distance_threshold

        #Init model
        model_path = "Model-V0.xml"
        fullpath = os.path.join(
            os.path.dirname(__file__), "../assets/UR5", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        #Init simulation
        self.robot = mujoco.MjModel.from_xml_path(fullpath)
        self.sim = mujoco.MjData(self.robot)

        #config actuators
        self.init_qpos = [0.2, 1.8, 1.8 ,0.3, 0.7, 0.5]
        self.init_qvel = [0,0,0,0,0,0]
        self.num_actuators = len(self.sim.ctrl)
        self.qpos_bounds = np.array(((-1, 1), (0, 2), (0, 2), (0, 2), (0, 2), (-1, 1)), dtype=object) # rango de articulaciones

        #config action space
        bounds = self.robot.actuator_ctrlrange.copy().astype(np.float16)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high,dtype=np.float16)

        #config observation space
        observation = self.get_observation()
        self.observation_space = convert_observation_to_space(observation)

        #config target
        self.target_bounds = np.array(((-0.3, 0.1), (-0.3, 0.3), (0.45, 0.5)), dtype=object) #limites del target a alcanzar

        self.reset()

    def reset(self):

        #nitialize the position of a random object to start the episode.
        self.reset_target()

        self.sim.qpos[:] = self.init_qpos
        self.sim.qvel[:] = self.init_qvel

        mujoco.mj_forward(self.robot, self.sim)

        return self.get_observation()


    def step(self, action):

        #Init variables
        done = False
        reward = 0
        action = np.clip(action, self.action_space.low, self.action_space.high) # I make sure that we don't change the action outside the limits of the target

        self.do_simulation(action,self.simulation_frames)

        observation = self.get_observation()

        reward = self.compute_reward(observation, action)

        if (reward == 1):
            done = True

        info = self.get_info(observation)

        return observation, reward, done, info

    ##### usefull methods ######

    def get_observation(self):
        """
        This function returns the position and velocity of the joints and 
        the xyz position of the claw.
        """

        left_finger = self.sim.body("left_inner_finger").xpos.astype(np.float16)
        right_finger = self.sim.body("right_inner_finger").xpos.astype(np.float16)

        gripper_position = ((left_finger[0] + right_finger[0])/2, (left_finger[1] + right_finger[1])/2, (left_finger[2] + right_finger[2])/2)

        target_position = self.sim.geom("target").xpos
        joints_position = self.sim.qpos.flat.copy().astype(np.float16)
        joints_velocity = self.sim.qvel.flat.copy().astype(np.float16)

        observation = np.concatenate(
            (gripper_position, target_position, joints_position, joints_velocity)
        )

        return observation


    def reset_target(self):
        """
        This function resets the goal position randomly.
        """

        self.goal = np.random.rand(3) * (self.target_bounds[:, 1] -
                                         self.target_bounds[:, 0]
                                         ) + self.target_bounds[:, 0]
        
        self.sim.geom("target").xpos = self.goal

    def do_simulation(self, ctrl, n_frames):
        """
        This function allows applying control on n simulation frames. These 
        steps are different from the agent's steps. The simulation frames are 
        simulation steps using a control action.        
        """

        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("dimensión  de las acción no concuerda con el controlador")

        self.sim.ctrl[:] = ctrl

        for _ in range(n_frames):
            mujoco.mj_step(self.robot,self.sim)

    def compute_reward(self, state, action):
        """
        This function compute the reward
        """

        gripper_position = np.array([state[0], state[1], state[2]])
        target_position = self.goal.copy().astype(np.float16)

        distance_norm = np.linalg.norm(target_position - gripper_position).astype(np.float16)

        action_norm = np.linalg.norm(action).astype(np.float16)

        if (distance_norm < self.distance_threshold):
            return 1

        return (-distance_norm - self.C_a * action_norm).astype(np.float16)

    def get_info(self, observation):
        """
        This function returns info

        ### description
            --"gripper_position": xyz position of the end effector.
            --"target_position": target position.
            --"j_position": position of the joints.
            --"j_velocity": velocity of the joints.
            --"dist": distance between the end effector and the goal.
        """

        gripper_position = self.sim.body("ee_link").xpos.astype(np.float16)
        target_position = self.sim.geom("target").xpos.astype(np.float16)

        info = {
            'gripper_position': gripper_position,
            'target_position': target_position,
            'dist': np.linalg.norm(target_position - gripper_position).astype(np.float16),
            'observation': self.get_observation(),
            'j_position': self.sim.qpos.flat.copy().astype(np.float16),
            'j_velocity': self.sim.qvel.flat.copy().astype(np.float16),
        }

        return info
