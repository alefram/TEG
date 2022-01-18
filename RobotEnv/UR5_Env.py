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
        pass

    def step(self, action):

        #aplicar

        return self.observations(), reward, done


    def reset(self):

        #obtener la posicion inicial del robot

        #inicializar las posiciones  y velocidades de las articulaciones

        return self.observations()


    def render(self,mode='human', camera=None):
        rgb = self.sim.render(width=self.width, height = self.height, camera_name=camera)
        return rgb


    def close(self):
        pass


    ##### funciones utiles ######

    def observations(self):

        pass
