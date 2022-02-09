import mujoco_py
import numpy as  np
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class UR5_EnvTest():
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


        # configurar actuadores
        self.init_qpos = [0,1,0,1,1,0] #testing
        self.init_qvel = [0,0,0,0,0,0]
        self.num_actuators = len(self.sim.data.ctrl)


        #configurar los espacio de acción


        #configurar el target


        self.seed()
        self.reset()

    def reset(self):

        #inicializar la posición de un objeto eleatorio para iniciar el episodio
        # self.reset_target()

        #inicializar las posiciones  y velocidades de las articulaciones
        # nq: numero de coordenadas generalizadas
        # nv: numero de grados de libertad
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
        # TODO: hacer sistema de recompenza

        #crear las acciones posibles del agente
        for i in range(self.num_actuators):
            controller[i] = 1


        # aplicar control en paso de simulación
        # estos pasos son distintos de los pasos del agente
        # los simulation frames son los pasos que el agente evita computar
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
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
