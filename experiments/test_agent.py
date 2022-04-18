"""
programa que metiendo seleccionando el
agente a usar y una posicion de la garra deseada,
simule ese movimiento.

"""

#importar librerias
import os
import mujoco_py
import numpy as np
import torch
import argparse
import time

# agregar el agente de input
parsers = argparse.ArgumentParser()
parsers.add_argument('-model', '--model', required=True, help='agente a usar')
parsers.add_argument('-robot', '--robot', required=True, help='modelo del robot')
parsers.add_argument('-x', '--x', required=True, help='x target position', type=float)
parsers.add_argument('-y', '--y', required=True, help='y target position', type=float)
parsers.add_argument('-z', '--z', required=True, help='z target position', type=float)

args = vars(parsers.parse_args())


# crear clase controlador inteligente
class Manipulator_Agent():
    def __init__(self, model, robot, frames):

        self.robot = mujoco_py.load_model_from_path("./RobotEnv/assets" + robot)
        self.sim = mujoco_py.MjSim(self.robot)
        self.model = torch.load("./agents" + model)
        self.simulation_frames = frames

        self.init_qpos = [-0.1, 0, 1 ,0.3, 0, 0]
        self.init_qvel = [0,0,0,0,0,0]
        self.sim.data.qpos[:] = self.init_qpos
        self.sim.data.qvel[:] = self.init_qvel

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


    def move_to(self, target):
        """mover la posición de la garra hacia el target"""

        assert target.size == 3

        simulation_positions = self.sim.model.geom_pos.copy()
        simulation_positions[1] = target
        self.sim.model.geom_pos[:] = simulation_positions

        action = self.model.act(torch.as_tensor(self.observe(), dtype=torch.float32))
        self.sim.data.ctrl[:] = action

        for _ in range(self.simulation_frames):
            self.sim.step()

        print("target", self.sim.data.get_geom_xpos("target").astype(np.float32))
        print("garra", self.sim.data.get_body_xpos("left_inner_finger").astype(np.float32))
        print(action)



def main():

    controller = Manipulator_Agent(args['model'], args['robot'], 4)
    viewer = mujoco_py.MjViewer(controller.sim)

    target = np.array([args['x'], args['y'], args['z']])


    for t in range(10000):

        viewer.render()
                
        controller.move_to(target)

        if t == 500:
            target = np.array([-0.1, -0.2, 0.5])
            time.sleep(0.05)

        if t == 1000:
            target = np.array([-0.1, 0.2, 0.5])
            time.sleep(0.05)

        if t == 1500:
            target = np.array([0.0, 0.0, 0.5])
            time.sleep(0.05)



if __name__ == '__main__':
    main()
