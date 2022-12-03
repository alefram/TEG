import os
import time
import torch
import numpy as np
import mujoco_py

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

        done = False
        steps = 0

        self.sim.forward()

        for t in range(timer):

            #mostrar la simulación
            if self.render:
                self.viewer.render()

            #observar
            observation = self.observe()

            # calcular la distancia entre el target y el efector final
            gripper_position = np.array([observation[0], observation[1], observation[2]])
            distance_norm = np.linalg.norm(target - gripper_position).astype(np.float32)
            print(gripper_position)

            # aplicar acción de control
            action = self.model.act(torch.as_tensor(observation, dtype=torch.float32))
            self.sim.data.ctrl[:] = action
            self.sim.step()

            # si la distancia entre el target y el limite es menor aplicar 
            # torque constante
            if (distance_norm <= distance_threshold):

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

                done = True
                # self.sim.step()
                steps += 1

                print('resuelto en:', steps, "pasos", steps*0.002, "seg")
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

            steps += 1


            if steps == timer:
                print('no se pudo alcanzar el target en:', steps, "pasos", steps*0.002, 
                    "seg")

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

        return position, qpos, control, done, steps

    def reset(self):

        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = self.init_qvel

        self.sim.forward()

    def close(self):
        if self.viewer is not None:
            self.viewer = None
        
    def change_agent(self,agent):
        fullpath = "/home/alexis/Documentos/repos/TEG/agents/"+ \
                agent + "/pyt_save/model.pt"

        self.model = torch.load(fullpath)
