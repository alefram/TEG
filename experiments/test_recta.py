"""
programa que metiendo los parametros de una recta el brazo dibuja la recta
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
from RobotEnv.tools.logger import Logger
import matplotlib as mpl
import matplotlib.pyplot as plt

# agregar el agente de input
parsers = argparse.ArgumentParser()
parsers.add_argument('-model', '--model', required=True, help='agente a usar')
parsers.add_argument('-robot', '--robot', required=True, help='modelo del robot')   
args = vars(parsers.parse_args())

#TODO: ajustar simulación para medir en paso de tiempo del state de la simulación

def main():

    #crear la simulation
    simulation = create_simulation(args['robot'])

    # crear controlador
    controller = Manipulator_Agent(args['model'], simulation, 4, render=True)

    # #crear ventana de visualización
    # viewer = mujoco_py.MjViewer(simulation)
    
    #TODO: crear trayectorias de recta y circulo
    #vectores
    output = []
    output2 = []
    output3 = []
    time_list = []
    x = [i/100 for i in range(-40,40)]
    y = x 


    target = np.array([0.1, 0.1, 0.5])


    for episode in range(len(x)):

        controller.move_to(np.array([x[episode], y[episode], 0.5]), timer=100)
        output.append(simulation.data.get_body_xpos("left_inner_finger")[0].astype(np.float32))
        output2.append(simulation.data.get_body_xpos("left_inner_finger")[1].astype(np.float32))
        output3.append(simulation.data.ctrl[5])
        time_list.append(episode)

    logger = Logger()

    # logger.plot_trajectory(x,y)
    logger.plot_error(time_list, error_a=y, error_b=output2)

    logger.show()

if __name__ == '__main__':
    main()
