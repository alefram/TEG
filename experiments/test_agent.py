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
from RobotEnv.tools.simulation import create_simulation
from RobotEnv.tools.controllers import Manipulator_Agent

# agregar el agente de input
parsers = argparse.ArgumentParser()
parsers.add_argument('-model', '--model', required=True, help='agente a usar')
parsers.add_argument('-robot', '--robot', required=True, help='modelo del robot')
parsers.add_argument('-x', '--x', required=True, help='x target position', type=float)
parsers.add_argument('-y', '--y', required=True, help='y target position', type=float)
parsers.add_argument('-z', '--z', required=True, help='z target position', type=float)

args = vars(parsers.parse_args())


def main():

    #crear la simulation
    simulation = create_simulation(args['robot'])

    # crear controlador
    controller = Manipulator_Agent(args['model'], simulation, 4)

    #crear ventana de visualización
    viewer = mujoco_py.MjViewer(simulation)
    
    #definir target
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
