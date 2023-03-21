"""experimento 1 """

from RobotEnv.envs.UR5_Env import UR5_EnvTest
from RobotEnv.tools import simulation
from RobotEnv.tools import controllers
from RobotEnv.tools.ik import inverse_kinematics
import numpy as np
import os
import argparse
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from dm_control import mujoco

# construir los inputs del usuario

parser = argparse.ArgumentParser()

parser.add_argument("--agent1", help="agente 1")
parser.add_argument("--agent2", help="agente 2")
parser.add_argument("--dist", help="distancia minima para lograr la tarea", \
                    type=float)
parser.add_argument("--render", help="mostrar simulación")
parser.add_argument("-t", "--timer", \
                    help="tiempo de duración del controlador ajustando",type=int)

args = parser.parse_args()

dist = args.dist
render = args.render
geom_pos = 1
timer = args.timer
agent = args.agent1
agent2 = args.agent2
PHYSICS_PATH = "../RobotEnv/assets/UR5/robotModelV3.xml"

sim = simulation.create_simulation("robotModelV3.xml")


# loop de recorido de la trayectoria
def main():

    #circulo
    theta = np.linspace(0, 2*np.pi, 15)
    r = np.sqrt(0.01)
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    #datos
    datax = []
    datay = []
    datax2 = []
    datay2 = []

    q_wrist1 =   []
    q_wrist2 =   []
    q_wrist3 =   []
    q_base   =   []
    q_shoulder = []
    q_elbow    = []
    q2_wrist1 =   []
    q2_wrist2 =   []
    q2_wrist3 =   []
    q2_base   =   []
    q2_shoulder = []
    q2_elbow    = []

    ctrl_wrist1 =   []
    ctrl_wrist2 =   []
    ctrl_wrist3 =   []
    ctrl_base   =   []
    ctrl_shoulder = []
    ctrl_elbow    = []
    ctrl2_wrist1 =   []
    ctrl2_wrist2 =   []
    ctrl2_wrist3 =   []
    ctrl2_base   =   []
    ctrl2_shoulder = []
    ctrl2_elbow    = []

    total_steps1 = 0
    total_steps2 = 0

    controller = controllers.Manipulator_Agent(agent, sim, render=render)

    for i in range(len(theta)):

        target = np.array([x[i]-0.1, y[i], 0.5])
        simulation.post_target(sim, target, geom_pos)

        pos, qpos, control, d, steps = controller.move_to(np.array(target),
                                    distance_threshold=dist,
                                    timer=timer)

        obs = controller.observe()
        datax.append(obs[0])
        datay.append(obs[1])

        q_base += qpos["base_link"]
        q_elbow += qpos["elbow_link"]
        q_shoulder += qpos["shoulder_link"]
        q_wrist1 += qpos["wrist_1_link"]
        q_wrist2 += qpos["wrist_2_link"]
        q_wrist3 += qpos["wrist_3_link"]

        ctrl_base += control["base_link"]
        ctrl_elbow += control["elbow_link"]
        ctrl_shoulder += control["shoulder_link"]
        ctrl_wrist1 += control["wrist_1_link"]
        ctrl_wrist2 += control["wrist_2_link"]
        ctrl_wrist3 += control["wrist_3_link"]

        total_steps1 += steps

    controller.change_agent(agent2)
    controller.reset()

    for j in range(len(theta)):

        target = np.array([x[j]-0.1, y[j], 0.5])
        simulation.post_target(sim, target, geom_pos)

        pos2, qpos2, control2, d2, steps2 = controller.move_to(np.array(target),
                                    distance_threshold=dist,
                                    timer=timer)

        obs2 = controller.observe()
        datax2.append(obs2[0])
        datay2.append(obs2[1])

        q2_base += qpos2["base_link"]
        q2_elbow += qpos2["elbow_link"]
        q2_shoulder += qpos2["shoulder_link"]
        q2_wrist1 += qpos2["wrist_1_link"]
        q2_wrist2 += qpos2["wrist_2_link"]
        q2_wrist3 += qpos2["wrist_3_link"]

        ctrl2_base += control2["base_link"]
        ctrl2_elbow += control2["elbow_link"]
        ctrl2_shoulder += control2["shoulder_link"]
        ctrl2_wrist1 += control2["wrist_1_link"]
        ctrl2_wrist2 += control2["wrist_2_link"]
        ctrl2_wrist3 += control2["wrist_3_link"]

        total_steps2 += steps2


    controller.close()

# --------------------------------- RESULTADOS----------------------------------

# variable de ayuda

    steps_array = [i for i in range(total_steps1)]
    steps_array2 = [i for i in range(total_steps2)]

# grafica de la trayectoria que genera en comparación de la ideal.
    fig, ax = plt.subplots()
    ax.plot(x, y, 0.5,'o', linestyle='-', label='objetivo')
    ax.plot(datax, datay, 0.5, 'v', linestyle='--' ,label="agente 1")
    ax.plot(datax2, datay2, 0.5, 'v', linestyle='-.' ,label="agente 2")
    ax.set_xlabel("posición x")
    ax.set_ylabel("posición y")
    ax.grid(True)
    ax.legend()

#TODO: colocar grafica de los angulos inferidos

    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(steps_array, q_base, linewidth=2.0, label="agente 1")
    ax2.plot(steps_array2, q2_base, linewidth=2.0, label="agente 2")
    ax2.set_xlabel("iteraciones")
    ax2.set_ylabel("angulo(rad) de base_link")
    ax2.legend()
    ax2.grid(True)

    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(steps_array, q_shoulder, linewidth=2.0, label="agente 1")
    ax3.plot(steps_array2, q2_shoulder, linewidth=2.0, label="agente 2")
    ax3.set_xlabel("iteraciones")
    ax3.set_ylabel("angulo(rad) de shoulder_link")
    ax3.legend()
    ax3.grid(True)

    fig4, ax4 = plt.subplots(1, 1)
    ax4.plot(steps_array, q_elbow, linewidth=2.0, label="agente 1")
    ax4.plot(steps_array2, q2_elbow, linewidth=2.0, label="agente 2")
    ax4.set_xlabel("iteraciones")
    ax4.set_ylabel("angulo(rad) de elbow_link")
    ax4.legend()
    ax4.grid(True)

    fig5, ax5 = plt.subplots(1, 1)
    ax5.plot(steps_array, q_wrist1, linewidth=2.0, label="agente 1")
    ax5.plot(steps_array2, q2_wrist1, linewidth=2.0, label="agente 2")
    ax5.set_xlabel("iteraciones")
    ax5.set_ylabel("angulo(rad) de wrist_1_link")
    ax5.legend()
    ax5.grid(True)

    fig6, ax6 = plt.subplots(1, 1)
    ax6.plot(steps_array, q_wrist2, linewidth=2.0, label="agente 1")
    ax6.plot(steps_array2, q2_wrist2, linewidth=2.0, label="agente 2")
    ax6.set_xlabel("iteraciones")
    ax6.set_ylabel("angulo(rad) de wrist_2_link")
    ax6.legend()
    ax6.grid(True)

    fig7, ax7 = plt.subplots(1, 1)
    ax7.plot(steps_array, q_wrist3, linewidth=2.0, label="agente 1")
    ax7.plot(steps_array2, q2_wrist3, linewidth=2.0, label="agente 2")
    ax7.set_xlabel("iteraciones")
    ax7.set_ylabel("angulo(rad) de wrist_3_link")
    ax7.legend()
    ax7.grid(True)  

# colocar grafica del torque aplicado. 
    figA, axA = plt.subplots()
    axA.plot(steps_array, ctrl_base, linewidth=2.0, label="agente 1")
    axA.plot(steps_array2, ctrl2_base, linewidth=2.0, label="agente 2")
    axA.set_xlabel("Iteraciones")
    axA.set_ylabel("torque(Nm) de base_link")
    axA.grid(True)
    axA.legend()

    figB, axB = plt.subplots()
    
    axB.plot(steps_array, ctrl_shoulder, linewidth=2.0, label="agente 1")
    axB.plot(steps_array2, ctrl2_shoulder, linewidth=2.0, label="agente 2")
    axB.set_xlabel("Iteraciones")
    axB.set_ylabel("torque(Nm) de shoulder_link")
    axB.grid(True)
    axB.legend()

    figC, axC = plt.subplots()

    axC.plot(steps_array, ctrl_elbow, linewidth=2.0, label="agente 1")
    axC.plot(steps_array2, ctrl2_elbow, linewidth=2.0, label="agente 2")
    axC.set_xlabel("Iteraciones")
    axC.set_ylabel("torque(Nm) de elbow_link")
    axC.grid(True)
    axC.legend()


    figD, axD = plt.subplots()

    axD.plot(steps_array, ctrl_wrist1, linewidth=2.0, label="agente 1")
    axD.plot(steps_array2, ctrl2_wrist1, linewidth=2.0, label="agente 2")
    axD.set_xlabel("Iteraciones")
    axD.set_ylabel("torque(Nm) de wrist_1_link")
    axD.grid(True)
    axD.legend()

    figE, axE = plt.subplots()

    axE.plot(steps_array, ctrl_wrist2, linewidth=2.0, label="agente 1")
    axE.plot(steps_array2, ctrl2_wrist2, linewidth=2.0, label="agente 2")
    axE.set_xlabel("Iteraciones")
    axE.set_ylabel("torque(Nm) de wrist_2_link")
    axE.grid(True)
    axE.legend()

    figF, axF = plt.subplots()

    axF.plot(steps_array, ctrl_wrist3, linewidth=2.0, label="agente 1")
    axF.plot(steps_array2, ctrl2_wrist3, linewidth=2.0, label="agente 2")
    axF.set_xlabel("Iteraciones")
    axF.set_ylabel("torque(Nm) wrist_3_link")
    axF.grid(True)
    axF.legend()

# calculo del error
    y = np.array(y)
    x = np.array(x)

    datay = np.array(datay)
    datax = np.array(datax)
    datax2 = np.array(datax2)
    datay2 = np.array(datay2)

    msey = np.square(np.subtract(y,datay)).mean()
    msex = np.square(np.subtract(x,datax)).mean()
    
    msey2 = np.square(np.subtract(y,datay2)).mean()
    msex2 = np.square(np.subtract(x,datax2)).mean()

    print("error cuadratico medio en x del agente 1:", msex)
    print("error cuadratico medio en y del agente 1:", msey)
    print("error cuadratico medio en x del agente 2:", msex2)
    print("error cuadratico medio en y del agente 2:", msey2)

# guardar
    fig.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-trayectoria2D.png")
    fig2.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-anguloBase.png")
    fig3.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-anguloShoulder.png")
    fig4.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-anguloElbow.png")
    fig5.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-anguloWrist1.png")
    fig6.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-anguloWrist2.png")
    fig7.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-anguloWrist3.png")
    figA.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-torqueBase.png")
    figB.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-torqueShoulder.png")
    figC.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-torqueElbow.png")
    figD.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-torqueWrist1.png")
    figE.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-torqueWrist2.png")
    figF.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E2-torqueWrist3.png")

    plt.show()


if __name__ == "__main__":
    main()
