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
    def __init__(self, model_path, simulation, render):

        self.sim = simulation

        fullpath = "/home/alexis/Documentos/repos/TEG/agents/"+ \
                    model_path + "/pyt_save/model.pt"

        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.model = torch.load(fullpath)

        self.init_qpos = [0.2, 1.8, 1.8 ,0.3, 0.7, 0.5]
        self.init_qvel = [0,0,0,0,0,0]
        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = self.init_qvel

        self.render = render

        if self.render:
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.reset()

    def observe(self):
        """observar mi entorno"""

        left_finger = self.sim.data.get_body_xpos("left_inner_finger").astype(np.float32)
        right_finger = self.sim.data.get_body_xpos("right_inner_finger").astype(np.float32)
        gripper_position = ((left_finger[0] + right_finger[0])/2,
                            (left_finger[1] + right_finger[1])/2,
                            (left_finger[2] + right_finger[2])/2)

        # gripper_position = self.sim.data.get_body_xpos("left_inner_finger").astype(np.float32)
        target_position = self.sim.data.get_geom_xpos("target").astype(np.float32)
        joints_position = self.sim.data.qpos.flat.copy().astype(np.float32)
        joints_velocity = self.sim.data.qvel.flat.copy().astype(np.float32)

        observation = np.concatenate(
            (gripper_position, target_position, joints_position, joints_velocity)
        )

        return observation

    def move_to(self, target, distance_threshold=0.05, timer=100):
        """mover la posición de la garra hacia el target"""

        assert target.size == 3

        data_x = []
        data_y = []
        data_z = []

        qpos1 = []
        qpos2 = []
        qpos3 = []
        qpos4 = []
        qpos5 = []
        qpos6 = []

        control1 = []
        control2 = []
        control3 = []
        control4 = []
        control5 = []
        control6 = []

        obs = []

        # self.sim.forward()

        for t in range(timer):

            #mostrar la simulación
            if self.render:
                self.viewer.render()

            #observar
            observation = self.observe()

            # calcular la distancia entre el target y el efector final
            gripper_position = np.array([observation[0], observation[1], observation[2]])
            target_position = np.array([observation[3], observation[4], observation[5]])
            distance_norm = np.linalg.norm(target_position - gripper_position).astype(np.float32)

            # aplicar acción de control
            action = self.model.act(torch.as_tensor(observation, dtype=torch.float32))
            self.sim.data.ctrl[:] = action

            self.sim.step()

            # si la distancia entre el target y el limite es menor aplicar torque constante
            if (distance_norm < distance_threshold):

                #guardar data
                qpos1.append(observation[6])
                qpos2.append(observation[7])
                qpos3.append(observation[8])
                qpos4.append(observation[9])
                qpos5.append(observation[10])
                qpos6.append(observation[11])

                control1.append(action[0])
                control2.append(action[1])
                control3.append(action[2])
                control4.append(action[3])
                control5.append(action[4])
                control6.append(action[5])

                data_x.append(gripper_position[0])
                data_y.append(gripper_position[1])
                data_z.append(gripper_position[2])

                obs.append(observation)

                print('resuelto en:', t, "pasos", t*0.002, "seg")
                break

            #guardar data
            qpos1.append(observation[6])
            qpos2.append(observation[7])
            qpos3.append(observation[8])
            qpos4.append(observation[9])
            qpos5.append(observation[10])
            qpos6.append(observation[11])

            control1.append(action[0])
            control2.append(action[1])
            control3.append(action[2])
            control4.append(action[3])
            control5.append(action[4])
            control6.append(action[5])

            data_x.append(gripper_position[0])
            data_y.append(gripper_position[1])
            data_z.append(gripper_position[2])

            obs.append(observation)

            if t == timer-1:
                print('no se pudo alcanzar el target en:', t, "pasos", t*0.002, "seg")

        qpos = {
            "base_link": qpos1,
            "shoulder_link": qpos2,
            "elbow_link": qpos3,
            "wrist_1_link": qpos4,
            "wrist_2_link": qpos5,
            "wrist_3_link": qpos6
        }

        control = {
            "base_link": control1,
            "shoulder_link": control2,
            "elbow_link": control3,
            "wrist_1_link": control4,
            "wrist_2_link": control5,
            "wrist_3_link": control6
        }

        position = {
            "pos_x": data_x,
            "pos_y": data_y,
            "pos_z": data_z,
        }

        return position, qpos, control, obs

    def reset(self):

        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = self.init_qvel

        self.sim.forward()

    def close(self):
        if self.viewer is not None:
            self.viewer = None
