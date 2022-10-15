"""experimento 1 """

from RobotEnv.envs.UR5_Env import UR5_EnvTest
from RobotEnv.tools import simulation
from RobotEnv.tools import controllers
import numpy as np
import os
import argparse
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt 
from RobotEnv.tools.ik import inverse_kinematics
from dm_control import mujoco

# construir los inputs del usuario
parser = argparse.ArgumentParser()

parser.add_argument("--agent", help="selecionar agente")
parser.add_argument("--dist", 
                    help="distancia minima para lograr la tarea", \
                    type=float)
parser.add_argument("--render", help="mostrar simulación")
parser.add_argument("-t", "--timer", 
                    help="tiempo de duración del controlador ajustando", \
                    type=int)
parser.add_argument("-i", "--episodes", help="episodios", type=int)

args = parser.parse_args()

dist = args.dist
render = args.render
geom_pos = 1
timer = args.timer
episodes = args.episodes
agent = args.agent
PHYSICS_PATH = "../RobotEnv/assets/UR5/robotModelV3.xml"


sim = simulation.create_simulation("robotModelV3.xml")
controller = controllers.Manipulator_Agent(agent, sim, render=render)


def main():

    # recta
#    x =  [i/100 for i in range(-2,2)]
    x = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 
         0.0, -0.05, -0.1, -0.15, -0.2]
    y = x

    #data
    posx = []
    posy = []
    posz = []
    datax = []
    datay = []
    targets = []

    for i in range(episodes):
        print('---------------------')
        print("episodio", i)
        print('---------------------')

        controller.reset()

        for i in range(len(x)):
            
            target = np.array([x[i], y[i], 0.45])
            targets.append(target)
            simulation.post_target(sim, target, geom_pos)

            pos, qpos, control, d, steps, = controller.move_to(np.array(target), 
                                                distance_threshold=dist, 
                                                timer=timer+1
                                                )
            obs = controller.observe()
            datax.append(obs[0])
            datay.append(obs[1])
            posx.extend(pos["pos_x"])
            posy.extend(pos["pos_y"])
            posz.extend(pos["pos_z"])



# ------------------------------RESULTADOS--------------------------------------


# variable de ayuda

    print("targets:", targets[0])

    # crear el modelo de dm_control del robot
    physics = mujoco.Physics.from_xml_path(PHYSICS_PATH)

    # hacer cinematica inversa
    _joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    _tol = 1.2e-14
    _max_steps = timer
    _max_resets = 10
    _inplace = [False, True]
    _target = np.array([target[0], target[1], target[2]])
    _site = "test"

    steps_array = [i for i in range(steps)]
    targetx = [np.array(target[0]) for i in range(steps)]
    targety = [np.array(target[1]) for i in range(steps)]
    targetz = [np.array(target[2]) for i in range(steps)]


# colocar la grafica de la trayectoria que genera en comparación de la ideal.  
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, 0.5,'o', linestyle="-." , label='objetivo')
    # ax.plot(posx, posy, 0.5, linestyle="-.", label="robot trayectory")
    ax.plot(datax, datay, 0.5, 'v', linestyle="--", label="trayectoria")
    ax.legend()

# TODO: colocar grafica de los angulos inferidos.


    fig2, ax2 = plt.subplots(1, 1)
    reference = [result.qpos[0] for i in range(steps)]
    ax2.set_title("Angulo aplicado por el actuador base_link")
    ax2.plot(steps_array, qpos["base_link"], linewidth=2.0, label="trayectoria")
    # ax2.plot(steps_array, reference, linewidth=2.0, label="objetivo",
    #               linestyle="--", color="gray")
    ax2.set_xlabel("iteraciones")
    ax2.set_ylabel("angulo(rad)")
    ax2.legend()
    ax2.grid(True)

    fig3, ax3 = plt.subplots(1, 1)
    reference = [result.qpos[1] for i in range(steps)]
    ax3.set_title("Angulo aplicado por el actuador shoulder_link")
    ax3.plot(steps_array, qpos["shoulder_link"], linewidth=2.0, label="trayectoria")
    # ax3.plot(steps_array, reference, linewidth=2.0, label="objetivo",
    #               linestyle="--", color="gray")
    ax3.set_xlabel("iteraciones")
    ax3.set_ylabel("angulo(rad)")
    ax3.legend()
    ax3.grid(True)

    fig4, ax4 = plt.subplots(1, 1)
    reference = [result.qpos[2] for i in range(steps)]
    ax4.set_title("Angulo aplicado por el actuador elbow_link")
    ax4.plot(steps_array, qpos["elbow_link"], linewidth=2.0, label="trayectoria")
    # ax4.plot(steps_array, reference, linewidth=2.0, label="objetivo",
    #               linestyle="--", color="gray")
    ax4.set_xlabel("iteraciones")
    ax4.set_ylabel("angulo(rad)")
    ax4.legend()
    ax4.grid(True)

    fig5, ax5 = plt.subplots(1, 1)
    reference = [result.qpos[3] for i in range(steps)]
    ax5.set_title("Angulo aplicado por el actuador wrist_1_link")
    ax5.plot(steps_array, qpos["wrist_1_link"], linewidth=2.0, label="trayectoria")
    # ax5.plot(steps_array, reference, linewidth=2.0, label="objetivo",
    #               linestyle="--", color="gray")
    ax5.set_xlabel("iteraciones")
    ax5.set_ylabel("angulo(rad)")
    ax5.legend()
    ax5.grid(True)

    fig6, ax6 = plt.subplots(1, 1)
    reference = [result.qpos[4] for i in range(steps)]
    ax6.set_title("Angulo aplicado por el actuador wrist_2_link")
    ax6.plot(steps_array, qpos["wrist_2_link"], linewidth=2.0, label="trayectoria")
    # ax6.plot(steps_array, reference, linewidth=2.0, label="objetivo",
    #               linestyle="--", color="gray")
    ax6.set_xlabel("iteraciones")
    ax6.set_ylabel("angulo(rad)")
    ax6.legend()
    ax6.grid(True)

    fig7, ax7 = plt.subplots(1, 1)
    reference = [result.qpos[5] for i in range(steps)]
    ax7.set_title("Angulo aplicado por el actuador wrist_3_link")
    ax7.plot(steps_array, qpos["wrist_3_link"], linewidth=2.0, label="trayectoria")
    # ax7.plot(steps_array, reference, linewidth=2.0, label="objetivo",
    #               linestyle="--", color="gray")
    ax7.set_xlabel("iteraciones")
    ax7.set_ylabel("angulo(rad)")
    ax7.legend()
    ax7.grid(True)

# TODO: colocar grafica del torque aplicado. 
# TODO: colocar el error entre la trayectoria deseada y la generada.

# calculo del error
    y = np.array(y)
    datay = np.array(datay)
    mse = np.square(np.subtract(y,datay)).mean()
    print("error cuadratico medio:", mse)

    plt.show()

if __name__ == "__main__":
    main()
