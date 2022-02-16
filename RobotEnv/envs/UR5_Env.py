import mujoco_py
import numpy as  np
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class UR5_EnvTest(gym.Env):
    def __init__(self, rewarded_distance, simulation_frames, Gui):
        """
        argumentos:
            rewarded_distance: distancia recompenzada cuando te acercas a la distancia target

        """

        #inicializar configuraciones de la simulacion
        self.rewarded_distance = rewarded_distance
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
        print(self.action_space)
        

        #configurar el target
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
        controller = np.copy(self.sim.data.ctrl)


        #generar el sistema de recompenza


        #ejecutar accion del agente
        for i in range(self.num_actuators):
            controller[i] = 1


        # aplicar control en paso de simulación
        # estos pasos son distintos de los pasos del agente
        # los simulation frames son los pasos de simulación utilizando un controlador
        self.sim.data.ctrl[:] = controller
        for _ in range(self.simulation_frames):
            self.sim.step()


        # incrementar la recompenza a largo plazo
        self.acumulative_reward += 1



        return self.observations(), reward, done


    def render(self, camera=None):
        if self.Gui:
            self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None

    ##### funciones utiles ######

    def observations(self):
        """
        Esta función retorna la posicion y velocidad de las articulaciones
        """

        joints = np.concatenate(
            [self.sim.data.qpos.flat[:], self.sim.data.qvel.flat[:]]
        )

        return joints


    def reset_target(self):
        # Randomize goal position within specified bounds
        self.goal = np.random.rand(3) * (self.target_bounds[:, 1] -
                                         self.target_bounds[:, 0]
                                         ) + self.target_bounds[:, 0]
        geom_positions = self.sim.model.geom_pos.copy()
        prev_goal_location = geom_positions[1]


        geom_positions[1] = self.goal
        self.sim.model.geom_pos[:] = geom_positions


