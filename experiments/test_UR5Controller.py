"""
Programa que permita mover una posicion de la garra utilizando un controlador con pid y no un agente
"""

from RobotEnv.tools.controllers import Mujoco_controller
from RobotEnv.tools.simulation import CreateSimulation

def main():

    # crear simulacion
    simulation = CreateSimulation("UR5/robotModelV2.xml")


    # crear controlador
    controller = Mujoco_controller(simulation)
    

    # mover la posicion de la garra
    controller.move_to(np.array([0.2, 1.8, 1.8]))

if __name__ == '__main__':
    main()