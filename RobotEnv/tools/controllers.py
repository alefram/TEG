import os
import time
import torch
import numpy as np
import mujoco_py

#controlador  PID
class PID():
    """Controlador PID"""

    def __init__(self, P=0.0, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.0
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time
 
        self.reset()


    def reset(self):
        """Resetear coeficientes y calculos"""

        self.r_t = 0.0
        self.P = 0.0
        self.I = 0.0
        self.D = 0.0
        self.last_e_t = 0.0
        self.int_error = 0.0
        self.overshoot_guard = 20.0
        self.u_t = 0.0


    def update(self, y_t, current_time=None):
        """
            actualizar valores del PID dado la señal de salida

            u(t) = Kp e(t) + Ki integral(e(t)) + Kd d/dt (e(t)) = P + I + D
        """

        e_t = self.r_t - y_t

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_e_t = e_t - self.last_e_t

        if (delta_time >= self.sample_time):
            self.P = e_t
            self.I += e_t * delta_time

            if (self.I < -self.overshoot_guard):
                self.I = -self.overshoot_guard
            elif (self.I > self.overshoot_guard):
                self.I = self.ove

            self.D = 0.0

            if delta_time > 0:
                self.D = delta_e_t / delta_time

            #recordar ultimo valores para calculo final
            self.last_time = self.current_time
            self.last_e_t = e_t

            self.u_t = (self.Kp * self.P) + (self.Ki * self.I) + (self.Kd * self.D)


    def set_Kp(self, proportional_gain):
        """Ganancia proporcional"""
        self.Kp = proportional_gain

    def set_Ki(self, integral_gain):
        """Ganancia Integral"""
        self.Ki = integral_gain

    def set_Kd(self, derivative_gain):
        """Ganancia derivativa"""
        self.Kd = derivative_gain

    def setWindup(self, overshoot):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """

        self.overshoot_guard = overshoot

    def set_sample_time(self, sample_time):
        """tiempo de muestreo"""

        self.sample_time = sample_time


# crear clase controlador inteligente
class Manipulator_Agent():
    def __init__(self, model, simulation, frames, render):

        self.sim = simulation

        self.model = torch.load("./agents" + model)
        self.simulation_frames = frames

        self.init_qpos = [0.2, 1.8, 1.8 ,0.3, 0.7, 0.5]
        self.init_qvel = [0,0,0,0,0,0]
        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = self.init_qvel

        self.render = render

        if self.render:        
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.sim.forward()

    def observe(self):
        """observar mi entorno"""

        gripper_position = self.sim.data.get_body_xpos("left_inner_finger").astype(np.float32)
        target_position = self.sim.data.get_geom_xpos("target").astype(np.float32)
        joints_position = self.sim.data.qpos.flat.copy().astype(np.float32)
        joints_velocity = self.sim.data.qvel.flat.copy().astype(np.float32)

        observation = np.concatenate(
            (gripper_position, target_position, joints_position, joints_velocity)
        )

        return observation


    #TODO: agregar retorno de data de la posicion del efector final y articulaciones y acciones de control.
    def move_to(self, target, distance_threshold=0.05, stabilizer=0.01, timer=100):
        """mover la posición de la garra hacia el target"""

        assert target.size == 3
        left_finger = self.sim.data.get_body_xpos("left_inner_finger").astype(np.float32)
        right_finger = self.sim.data.get_body_xpos("right_inner_finger").astype(np.float32)

        gripper_position = ((left_finger[0] + right_finger[0])/2, (left_finger[1] + right_finger[1])/2, (left_finger[2] + right_finger[2])/2)
        target_position = self.sim.data.get_geom_xpos("target").astype(np.float32)
        distance_norm = np.linalg.norm(target_position - gripper_position).astype(np.float32)

        simulation_positions = self.sim.model.geom_pos.copy()
        simulation_positions[1] = target
        self.sim.model.geom_pos[:] = simulation_positions

        for t in range(timer):

            self.viewer.render()

            action = self.model.act(torch.as_tensor(self.observe(), dtype=torch.float32))
            self.sim.data.ctrl[:] = action

            if (distance_norm < distance_threshold):
                self.sim.data.ctrl[:] = stabilizer 

                
                for _ in range(self.simulation_frames):
                    self.sim.step()

                print('resuelto en:', t, "seg")
                break

            for _ in range(self.simulation_frames):
                self.sim.step()

# controlador clasico
class Mujoco_controller(object):
    """Controlador para un brazo manipulador usando mujoco y pid"""

    def __init__(self, simulation=None, frames=None):

        self.sim = simulation 
        self.reached_target = False
        self.q_current = np.zeros(len(self.sim.data.ctrl))
        self.q_reference = []
        self.create_control_list()


    def create_control_list(self):
        """crear lista de controladores"""

        self.control_list = []

        sample_time = 0.001

        self.control_list.append(PID(Kp=0.5, Ki=0.0, Kd=0.0, sample_time=sample_time)) #base 
        self.control_list.append(PID(Kp=0.5, Ki=0.0, Kd=0.0, sample_time=sample_time)) #shoulder
        self.control_list.append(PID(Kp=0.5, Ki=0.0, Kd=0.0, sample_time=sample_time)) #elbow
        self.control_list.append(PID(Kp=0.5, Ki=0.0, Kd=0.0, sample_time=sample_time)) #wrist1
        self.control_list.append(PID(Kp=0.5, Ki=0.0, Kd=0.0, sample_time=sample_time)) #wrist2
        self.control_list.append(PID(Kp=0.5, Ki=0.0, Kd=0.0, sample_time=sample_time)) #wrist3        
            
    # def move_to(self, target): 
    #     """mover la posición de la garra hacia el target"""

    #     assert target.size == 3

    #     # obtener posición del target y colocar en la simulacion
    #     simulation_positions = self.sim.model.geom_pos.copy()
    #     simulation_positions[1] = target
    #     self.sim.model.geom_pos[:] = simulation_positions

    #     # obtener el error
    #     self.q_reference = kinverse_kinematics(target, self.sim)
    #     self.q_current = self.sim.data.qpos.flat.copy().astype(np.float32)

    #     # aplicar pids
    #     pids = self.control_list

    #     for i in range(len(pids)):
    #         pids[i].update(self.q_current[i])

    #         self.sim.data.ctrl[i] = pids[i].u_t
            

    #     for _ in range(self.simulation_frames):
    #         self.sim.step()
        
    #     # verificar si la posición del target se alcanzó
    #     if np.linalg.norm(self.q_current - self.q_reference) < 0.05:
    #         self.reached_target = True
        
    #     return self.reached_target



