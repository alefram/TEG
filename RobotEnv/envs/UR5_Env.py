import mujoco_py
import numpy as  np
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class UR5_EnvTest():
    def __init__(self,width, height, rewarded_distance, control_magnitude, frame_skip, Gui):
        """
        argumentos:
            rewarded_distance: distancia recompenzada cuando te acercas a la distancia target
        """

        #inicializar configuraciones de la simulacion
        self.width = width
        self.height = height
        self.rewarded_distance = rewarded_distance
        self.acumulative_reward = 0
        self.frame_skip = frame_skip
        self.Gui = Gui

        #inicializar el modelo del robot
        self.robot = mujoco_py.load_model_from_path('RobotEnv/assets/UR5/robotModelV2.xml')
        self.sim = mujoco_py.MjSim(self.robot)

        if self.Gui:
            self.viewer = mujoco_py.MjViewer(self.sim)


        # inicializar la posicion y velocidad
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        #configurar actuadores
        self.actuator_bounds = self.sim.model.actuator_ctrlrange #limites del actuador
        self.actuator_low = self.actuator_bounds[:, 0]
        self.actuator_high = self.actuator_bounds[:, 1]
        self.actuator_controlRange = self.actuator_high - self.actuator_low
        self.actuators = len(self.actuator_low)

        #configurar los espacio de acción discretos
        self.control_values = self.actuator_controlRange * control_magnitude
        self.num_actions = 2
        self.action_space = [list(range(self.num_actions))] * self.actuators
        self.observation_space = ((0,), (height, width, 3), (height, width, 3))

        #configurar los limites de la posicion objetivo
        self.target_bounds = np.array(((0.4, 0.6), (0.1, 0.3), (0.2, 0.3)))
        self.target_reset_distance = 0.2

        self.seed()
        self.reset()

    def step(self, action):
        dist = np.zeros(2)
        done = False
        reward = 0
        controller = np.copy(self.sim.data.ctrl).flatten()

        #general el sistema de recompenza
        dist[0] = np.linalg.norm (
            self.sim.data.get_body_xpos("left_inner_finger") - self.goal
        )
        dist[1] = np.linalg.norm (
            self.sim.data.get_body_xpos("right_inner_finger") -  self.goal
        )

        if any(d < self.rewarded_distance for d in dist):
            reward = 1
            self.reset_target()

        #crear las acciones posibles del agente
        for i in range(self.actuators):
            '''
                0 = 0 velocidad
                1 = velocidad positiva
                2 = velocidad negativa
            '''
            if action[i] == 0:
                controller[i] = 0
            if action[i] == 1:
                controller[i] = self.control_values[i] / 2
            if action[i] == 2:
                controller[i] = -self.control_values[i] / 2




        #aplicar control en paso de simulación
        self.apply_control(controller)

        # incrementar la recompenza a largo plazo
        self.acumulative_reward += reward


        return self.observations(), reward, done


    def reset(self):

        #obtener la posicion inicial del robot
        qpos = self.init_qpos
        qvel = self.init_qvel

        #inicializar la posición de un objeto eleatorio para iniciar el episodio
        self.reset_target()

        #inicializar las posiciones  y velocidades de las articulaciones
        # nq: numero de coordenadas generalizadas
        # nv: numero de grados de libertad
        qpos_shape = int(''.join(map(str, qpos.shape)))
        qvel_shape = int(''.join(map(str, qvel.shape)))
        assert qpos_shape == (self.sim.model.nq) and qvel_shape == (self.sim.model.nv)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

        return self.observations()


    def render(self, camera=None):
        if self.Gui:
            self.viewer.render()
        else:
            rgb = self.sim.render(width=self.width, height = self.height, camera_name=camera)

            return rgb


    def close(self):
        pass


    ##### funciones utiles ######

    def observations(self):
        """
        Esta función retorna la posicion y velocidad de las articulaciones
        """

        joints = np.concatenate(
            [self.sim.data.qpos.flat[:], self.sim.data.qvel.flat[:]]
        )

        return joints

    def apply_control(self, control):
        """
        Esta función genera un paso de simulación, y actualiza la acción de control a los actuadores

        Argumentos:
        control: un vector del tamaño del numero de actuadores -- nuevo control enviado a los actuadores
        """

        control = np.min((control, self.actuator_high), axis=0)
        control = np.max((control, self.actuator_low), axis=0)

        self.sim.data.ctrl[:] = control

        for _ in range(self.frame_skip):
            self.sim.step()


    def reset_target(self):
        # crear una posición meta aleatorio
        self.goal = np.random.rand(3) * (self.target_bounds[:, 1] - self.target_bounds[:, 0]) + self.target_bounds[:, 0]

        geom_positions = self.sim.model.geom_pos.copy()
        prev_goal_location = geom_positions[1]

        while (np.linalg.norm(prev_goal_location - self.goal) < self.target_reset_distance):
            self.goal = np.random.rand(3) * (self.target_bounds[:, 1] - self.target_bounds[:, 0]) + self.target_bounds[:, 0]

        geom_positions[1] = self.goal
        self.sim.model.geom_pos[:] = geom_positions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
