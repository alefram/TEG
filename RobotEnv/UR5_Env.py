import mujoco_py
import numpy as  np
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class UR5_EnvTest():
    def __init__(self,width, height):

        #inicializar configuraciones de la simulacion
        self.width = width
        self.height = height

        #inicializar el modelo del robot
        self.robot = mujoco_py.load_model_from_path('assets/UR5/robotModel.xml')
        self.sim = mujoco_py.MjSim(self.robot)

        # inicializar la posicion y velocidad
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()


    def step(self, action):

        return self.observations(), reward, done


    def reset(self):

        #obtener la posicion inicial del robot
        qpos = self.init_qpos
        qvel = self.init_qvel

        #inicializar las posiciones  y velocidades de las articulaciones
        # nq: numero de coordenadas generalizadas
        # nv: numero de grados de libertad
        assert qpos.shape == (self.sim.model.nq) and qvel.shape == (self.sim.model.nv)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.foward()

        return self.observations()


    def render(self,mode='human', camera=None):
        rgb = self.sim.render(width=self.width, height = self.height, camera_name=camera)
        return rgb


    def close(self):
        pass


    ##### funciones utiles ######

    def observations(self):

        joints = np.concatenate(
            [self.sim.data.qpos.flat[:], self.sim.data.qvel.flat[:]]
        )

        return joints
