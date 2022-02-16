import mujoco_py
import numpy as  np
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class UR5_EnvTest(gym.Env):
    def __init__(self, simulation_frames, Gui):
        """
        argumentos:
            rewarded_distance: distancia recompenzada cuando te acercas a la distancia target

        """

        #inicializar configuraciones de la simulacion
        self.acumulative_reward = 0
        self.Gui = Gui
        self.simulation_frames = simulation_frames

        #inicializar el modelo del robot
        self.robot = mujoco_py.load_model_from_path('RobotEnv/assets/UR5/robotModelV2.xml')
        self.sim = mujoco_py.MjSim(self.robot)

        if self.Gui:
            self.viewer = mujoco_py.MjViewer(self.sim)


        #configurar actuadores
        self.init_qpos = [0,1,0,1,1,0]
        self.init_qvel = [0,0,0,0,0,0]
        self.num_actuators = len(self.sim.data.ctrl)


        #configurar los espacio de acción
        bounds = self.robot.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high,dtype=np.float32)
        

        #configurar el target
        
        #TODO: correguir que la posicion del target no pueda ser cerca o en la posicion donde esta el robot
        
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

        return self.observations()


    def step(self, action):

        #inicializar variables
        done = False
        reward = 0

        #TODO: generar el sistema de recompenza


        # aplicar control en paso de simulación
        # estos pasos son distintos de los pasos del agente
        # los simulation frames son los pasos de simulación utilizando un controlador
        self.do_simulation(action,self.simulation_frames)
        

        #TODO: agregar info
        info = {}

        return self.observations(), reward, done, info


    def render(self, camera=None):
        if self.Gui:
            self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None

    ##### funciones utiles ######

    def observations(self):
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
