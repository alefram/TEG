import mujoco_py
import numpy as  np
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class UR5_EnvTest(gym.Env):
    def __init__(self, model, simulation_frames, torque_control, distance_threshold, Gui):
        """
        argumentos:
            rewarded_distance: distancia recompenzada cuando te acercas a la distancia target

        """

        #inicializar configuraciones de la simulacion
        self.acumulative_reward = 0
        self.Gui = Gui
        self.simulation_frames = simulation_frames
        self.C_a = torque_control
        self.distance_threshold = distance_threshold
        self.robot_model = model

        #inicializar el modelo del robot
        self.robot = mujoco_py.load_model_from_path(robot_model)
        self.sim = mujoco_py.MjSim(self.robot)

        if self.Gui:
            self.viewer = mujoco_py.MjViewer(self.sim)


        #configurar actuadores
        self.init_qpos = [1, 1.8, 1.8 ,0.3,0.7,0.5]
        self.init_qvel = [0,0,0,0,0,0]
        self.num_actuators = len(self.sim.data.ctrl)


        #configurar los espacio de acción
        bounds = self.robot.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high,dtype=np.float32)
        

        #configurar el target
        geom_positions = self.sim.model.geom_pos.copy()
        self.target_position = geom_positions[1] #posicion del target
        
        #TODO: mejorar las limitaciones del target a el espacio de un cubo de tamaño x dentro del espacio de trabajo del brazo
        self.target_bounds = np.array(((-0.5, 0.5), (-0.5, 0.5), (0.45, 1))) #limites del target a alcanzar


        self.seed()
        self.reset()

    def reset(self):

        #inicializar la posición de un objeto eleatorio para iniciar el episodio
        self.reset_target()

        #inicializar las posiciones  y velocidades de las articulaciones
        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = self.init_qvel

        self.sim.forward()

        return self.get_observation()


    def step(self, action):

        #inicializar variables
        done = False
        reward = 0
        action = np.clip(action, self.action_space.low, self.action_space.high) # me aseguro que no cambiamos la accion fuera 

        # aplicar control en paso de simulación
        # estos pasos son distintos de los pasos del agente
        # los simulation frames son los pasos de simulación utilizando un controlador
        self.do_simulation(action,self.simulation_frames)
        
        # obtendo la observacion o el siguiente estado
        observation = self.get_observation()

        # obtengo la recompenza
        reward = self.compute_reward(observation, action)
        
        # verifico si la garra choca con el piso o recompenza -100 termina el episodio
        if (reward == -100):
            done = True
        
        # verifico que la garra este al menos de 5cm dando recompenza 1 y terminar el episodio
        # aqui se considera la lograda y terminada
        if (reward == 1):
            done = True
        
        info = self.get_info(observation)

        return observation, reward, done, info


    def render(self, camera=None):
        if self.Gui:
            self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None

    ##### funciones utiles ######

    def get_observation(self):
        '''
            Esta función retorna la posicion y velocidad de las articulaciones
        '''
        gripper_position = self.sim.data.get_body_xpos('ee_link')
        joints_position = self.sim.data.qpos.flat.copy()
        joints_velocity = self.sim.data.qvel.flat.copy()

        observation = np.concatenate(
            (gripper_position, joints_position, joints_velocity)
        )

        return observation


    def reset_target(self):
        # Randomize goal position within specified bounds
        self.goal = np.random.rand(3) * (self.target_bounds[:, 1] -
                                         self.target_bounds[:, 0]
                                         ) + self.target_bounds[:, 0]
        geom_positions = self.sim.model.geom_pos.copy()
        prev_goal_location = geom_positions[1]


        geom_positions[1] = self.goal
        self.sim.model.geom_pos[:] = geom_positions

    def do_simulation(self, ctrl, n_frames):
        '''
            aplicar el controlador a la simulación
        '''
        
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("dimesion  de las acción no concuerda con el controlador")
        
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def compute_reward(self, state, action):
        gripper_position = np.array([state[0], state[1], state[2]])
        target_position = self.target_position


        distance_norm = np.linalg.norm(target_position - gripper_position).astype(np.float32)
        action_norm = np.linalg.norm(action)
    
        if (gripper_position[2] <= 0.5):
            return -100

        if (distance_norm < self.distance_threshold):
            return 1

        return (-distance_norm - self.C_a * action_norm).astype(np.float32)

    def get_info(self, observation):
        gripper_position = self.sim.data.get_body_xpos('ee_link')

        info = {
            'gripper_position': self.sim.data.get_body_xpos('ee_link'),
            'j_position': self.sim.data.qpos.flat.copy().astype(np.float32),
            'j_velocity': self.sim.data.qvel.flat.copy().astype(np.float32),
            'dist': np.linalg.norm(self.target_position - gripper_position).astype(np.float32)
        }

        return info
