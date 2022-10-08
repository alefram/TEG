"""experimento 1 """

from RobotEnv.envs.UR5_Env import UR5_EnvTest
from RobotEnv.tools import simulation
from RobotEnv.tools import controllers
from RobotEnv.tools.logger import Logger
import numpy as np
import os
import argparse
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik
from dm_control.mujoco.wrapper import mjbindings

mjlib = mjbindings.mjlib


# construir los inputs del usuario

parser = argparse.ArgumentParser()

parser.add_argument("--agent", help="selecionar agente")
parser.add_argument("--dist", help="distancia minima para lograr la tarea", 
                    type=float)
parser.add_argument("--render", help="mostrar simulación")
parser.add_argument("-t", "--timer", 
                    help="tiempo de duración del controlador ajustando", 
                    type=int)
parser.add_argument("-i", "--episodes", help="episodios", type=int)

args = parser.parse_args()

dist = args.dist
render = args.render
target_bounds = np.array(((-0.3, 0.1), (-0.3, 0.3), (0.45, 0.5)), dtype=object)
geom_pos = 1
timer = args.timer
episodes = args.episodes
agent = args.agent

sim = simulation.create_simulation("robotModelV3.xml")

controller = controllers.Manipulator_Agent(agent, sim, render=render)


# loop de recorido de la trayectoria

def main():
    win = 0
    average_time = []
    position = {}
    qpos = {}
    control = {}

    for i in range(episodes):
        
        print('---------------------')
        print("episodio", i)
        print('---------------------')

        goal = simulation.random_target(target_bounds, geom_pos, sim)
        controller.reset()

        position, qpos, control, done, t, steps_array, target, target_position = \
                                            controller.move_to(
                                            np.array(goal), 
                                           distance_threshold=dist, timer=timer)
        
        average_time.append(t)

        if (done):
            win += 1

# ------------------------------RESULTADOS--------------------------------------

    print("-------------------------")
    print("Tasa de aciertos")
    print(win, "de", episodes)
    print("Promedio de tiempo")
    print((sum(average_time) / len(average_time))*0.002)
    print("-------------------------")

# TODO:hacer grafica de tasas de aciertos

# grafica de trayectoria final.
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection="3d")
    fig1.suptitle("Trayectoria para un objetivo")
    ax1.plot(position["pos_x"], position["pos_y"], position["pos_z"],
            linewidth=2.0, label="trayectoria")
    ax1.plot(target["x"], target["y"], target["z"], 'o', 
        linewidth=2.0, label="objetivo")
    ax1.set_xlabel("posición x")
    ax1.set_ylabel("posición y")
    ax1.set_zlabel("posición z")
    ax1.grid(True)
    ax1.legend()

    fig2, ax2 = plt.subplots(1,1)
    fig2.suptitle("Trayectoria del efector final en x por iteración")
    sup = target['x'] + np.array([dist for i in range(len(steps_array))])
    inf = target['x'] - np.array([dist for i in range(len(target['x']))])
    ax2.plot(steps_array, position['pos_x'], linewidth=2.0, label="trayectoria")
    ax2.plot(steps_array, target["x"], linestyle='--', linewidth=2.0, 
            label="objetivo")
    ax2.fill_between(steps_array, sup, inf, alpha=0.2)
    ax2.set_xlabel("tiempo(s)")
    ax2.set_ylabel("x(s)")
    ax2.grid(True)
    ax2.legend()

    fig3, ax3 = plt.subplots(1,1)
    fig3.suptitle("Trayectoria del efector final en y por iteración")
    sup2 = target['y'] + np.array([dist for i in range(len(steps_array))])
    inf2 = target['y'] - np.array([dist for i in range(len(target['y']))])
    ax3.plot(steps_array, position['pos_y'], linewidth=2.0, label="trayectoria")
    ax3.plot(steps_array, target["y"], linestyle='--', linewidth=2.0, 
            label="objetivo")
    ax3.fill_between(steps_array, sup2, inf2, alpha=0.2, label="umbral objetivo")
    ax3.set_xlabel("tiempo(s)")
    ax3.set_ylabel("y(s)")
    ax3.grid(True)
    ax3.legend()

    fig4, ax4 = plt.subplots(1,1)
    fig4.suptitle("Trayectoria del efector final en z por iteración")
    sup3 = target['z'] + np.array([dist for i in range(len(steps_array))])
    inf3 = target['z'] - np.array([dist for i in range(len(target['z']))])
    ax4.plot(steps_array, position['pos_z'], linewidth=2.0, label="trayectoria")
    ax4.plot(steps_array, target["z"], linestyle='--', linewidth=2.0, 
            label="objetivo")
    ax4.fill_between(steps_array, sup3, inf3, alpha=0.2, label="umbral objetivo")
    ax4.set_xlabel("tiempo(s)")
    ax4.set_ylabel("z(s)")
    ax4.grid(True)
    ax4.legend()

# graficar error de trayectoria cartesiana
    error_x = np.subtract(np.array(target["x"]), np.array(position["pos_x"]))
    error_y = np.subtract(np.array(target["y"]), np.array(position["pos_y"]))
    error_z = np.subtract(np.array(target["z"]), np.array(position["pos_z"]))
    reference = [ 0 for i in range(len(steps_array))]

    fig5, ax5 = plt.subplots(1,1)
    ax5.plot(steps_array, error_x, linewidth=2.0, label="error en eje x")
    ax5.plot(steps_array, error_y, linewidth=2.0, label="error en eje y")
    ax5.plot(steps_array, error_z, linewidth=2.0, label="error en eje z")
    ax5.plot(steps_array, reference, linewidth=2.0, label="referencia", 
            linestyle='--')
    ax5.set_xlabel("Iteraciones")
    ax5.set_ylabel("Error")
    ax5.grid(True)
    ax5.legend()

    fig5.suptitle("Error absoluto de la trayectoria")

#Colocar la grafica de los angulos aplicados durante la trayectoria.

    # crear el modelo de dm_control del robot

    physics = mujoco.Physics.from_xml_path("../assets/UR5/robotModelV3.xml")

    # hacer cinematica inversa

    _JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    _TOL = 1.2e-14
    _MAX_STEPS = timer
    _MAX_RESETS = 10
    _INPLACE = [False, True]
    _TARGET = np.array([target_position[0], target_position[1], target_position[2]])
    _SITE = "test"

    physics2 = physics.copy(share_model=True)

    result = ik.qpos_from_site_pose(
        physics=physics2,
        site_name=_SITE,
        target_pos=_TARGET,
        target_quat=None,
        joint_names=_JOINTS,
        tol=_TOL,
        max_steps=_MAX_STEPS,
        inplace=_INPLACE
    )

    print("angulos posibles:", result.qpos,"\n \n", "error:", result.err_norm)

    fig6, ax6 = plt.subplots(1, 1)
    reference = [result.qpos[0] for i in range(len(steps_array))]
    ax6.set_title("Angulo aplicado por el actuador base_link")
    ax6.plot(steps_array, qpos["base_link"], linewidth=2.0, label="trayectoria")
    ax6.plot(steps_array, reference, linewidth=2.0, label="objetivo",
                  linestyle="--", color="gray")
    ax6.set_xlabel("iteraciones")
    ax6.set_ylabel("angulo(rad)")
    ax6.legend()
    ax6.grid(True)

    fig7, ax7 = plt.subplots(1, 1)
    reference = [result.qpos[1] for i in range(len(steps_array))]
    ax7.set_title("Angulo aplicado por el actuador shoulder_link")
    ax7.plot(steps_array, qpos["shoulder_link"], linewidth=2.0, label="trayectoria")
    ax7.plot(steps_array, reference, linewidth=2.0, label="objetivo",
                  linestyle="--", color="gray")
    ax7.set_xlabel("iteraciones")
    ax7.set_ylabel("angulo(rad)")
    ax7.legend()
    ax7.grid(True)

    fig8, ax8 = plt.subplots(1, 1)
    reference = [result.qpos[2] for i in range(len(steps_array))]
    ax8.set_title("Angulo aplicado por el actuador elbow_link")
    ax8.plot(steps_array, qpos["elbow_link"], linewidth=2.0, 
            label="trayectoria")
    ax8.plot(steps_array, reference, linewidth=2.0, label="objetivo",
                  linestyle="--", color="gray")
    ax8.set_xlabel("iteraciones")
    ax8.set_ylabel("angulo(rad)")
    ax8.legend()
    ax8.grid(True)

    fig9, ax9 = plt.subplots(1, 1)
    reference = [result.qpos[3] for i in range(len(steps_array))]
    ax9.set_title("Angulo aplicado por el actuador wrist_1_link")
    ax9.plot(steps_array, qpos["wrist_1_link"], linewidth=2.0, 
            label="trayectoria")
    ax9.plot(steps_array, reference, linewidth=2.0, label="objetivo",
                  linestyle="--", color="gray")
    ax9.set_xlabel("iteraciones")
    ax9.set_ylabel("angulo(rad)")
    ax9.legend()
    ax9.grid(True)

    fig10, ax10 = plt.subplots(1, 1)
    reference = [result.qpos[4] for i in range(len(steps_array))]
    ax10.set_title("Angulo aplicado por el actuador wrist_2_link")
    ax10.plot(steps_array, qpos["wrist_2_link"], linewidth=2.0, 
            label="trayectoria")
    ax10.plot(steps_array, reference, linewidth=2.0, label="objetivo",
                  linestyle="--", color="gray")
    ax10.set_xlabel("iteraciones")
    ax10.set_ylabel("angulo(rad)")
    ax10.legend()
    ax10.grid(True)

    fig11, ax11 = plt.subplots(1, 1)
    reference = [result.qpos[5] for i in range(len(steps_array))]
    ax11.set_title("Angulo aplicado por el actuador wrist_3_link")
    ax11.plot(steps_array, qpos["wrist_3_link"], linewidth=2.0, 
            label="trayectoria")
    ax11.plot(steps_array, reference, linewidth=2.0, label="objetivo",
                  linestyle="--", color="gray")
    ax11.set_xlabel("iteraciones")
    ax11.set_ylabel("angulo(rad)")
    ax11.legend()
    ax11.grid(True)

# error de los angulos
    reference = [ 0 for i in range(len(steps_array))]
    result1 = [result.qpos[0] for i in range(len(steps_array))]
    result2 = [result.qpos[1] for i in range(len(steps_array))]
    result3 = [result.qpos[2] for i in range(len(steps_array))]
    result4 = [result.qpos[3] for i in range(len(steps_array))]
    result5 = [result.qpos[4] for i in range(len(steps_array))]
    result6 = [result.qpos[5] for i in range(len(steps_array))]

    error1 = np.subtract(result1, qpos["base_link"])
    error2 = np.subtract(result2, qpos["shoulder_link"])
    error3 = np.subtract(result3, qpos["elbow_link"])
    error4 = np.subtract(result4, qpos["wrist_1_link"])
    error5 = np.subtract(result5, qpos["wrist_2_link"])
    error6 = np.subtract(result6, qpos["wrist_3_link"])

    fig12, ax12 = plt.subplots(1,1, figsize=(10,6))
    ax12.plot(steps_array, error1, linewidth=2.0, label="base_link")
    ax12.plot(steps_array, error2, linewidth=2.0, label="shoulder_link")
    ax12.plot(steps_array, error3, linewidth=2.0, label="elbow_link")
    ax12.plot(steps_array, error4, linewidth=2.0, label="wrist_1_link")
    ax12.plot(steps_array, error5, linewidth=2.0, label="wrist_2_link")
    ax12.plot(steps_array, error6, linewidth=2.0, label="wrist_3_link")
    ax12.plot(steps_array, reference, linewidth=2.0, label="referencia", 
            linestyle='--')
    ax12.set_xlabel("Iteraciones")
    ax12.set_ylabel("Error")
    ax12.grid(True)
    ax12.legend()

# TODO:colocar la grafica del torque aplicado durante la trayectoria.

    # fig2, ax1 = plt.subplots(2, 3, figsize=(20,6), constrained_layout=True)

    # ax1[0,0].set_title("Torque aplicado por el actuador base_link")
    # ax1[0,0].plot(steps_array, control["base_link"], linewidth=2.0, color='red')
    # ax1[0,0].set_xlabel("tiempo(s)")
    # ax1[0,0].set_ylabel("torque(Nm)")
    # ax1[0,0].grid(True)

    # ax1[0,1].set_title("Torque aplicado por el actuador shoulder_link")
    # ax1[0,1].plot(steps_array, control["shoulder_link"], linewidth=2.0, color='blue')
    # ax1[0,1].set_xlabel("tiempo(s)")
    # ax1[0,1].set_ylabel("torque(Nm)")
    # ax1[0,1].grid(True)

    # ax1[1,0].set_title("Torque aplicado por el actuador elbow_link")
    # ax1[1,0].plot(steps_array, control["elbow_link"], linewidth=2.0, color='lightgrey')
    # ax1[1,0].set_xlabel("tiempo(s)")
    # ax1[1,0].set_ylabel("torque(Nm)")
    # ax1[1,0].grid(True)


    # ax1[1,1].set_title("Torque aplicado por el actuador wrist_1_link")
    # ax1[1,1].plot(steps_array, control["wrist_1_link"], linewidth=2.0, color='orange')
    # ax1[1,1].set_xlabel("tiempo(s)")
    # ax1[1,1].set_ylabel("torque(Nm)")
    # ax1[1,1].grid(True)


    # ax1[0,2].set_title("Torque aplicado por el actuador wrist_2_link")
    # ax1[0,2].plot(steps_array, control["wrist_2_link"], linewidth=2.0, color='green')
    # ax1[0,2].set_xlabel("tiempo(s)")
    # ax1[0,2].set_ylabel("torque(Nm)")
    # ax1[0,2].grid(True)

    # ax1[1,2].set_title("Torque aplicado por el actuador wrist_3_link")
    # ax1[1,2].plot(steps_array, control["wrist_3_link"], linewidth=2.0)
    # ax1[1,2].set_xlabel("tiempo(s)")
    # ax1[1,2].set_ylabel("torque(Nm)")
    # ax1[1,2].grid(True)


    # fig2.suptitle("Torque aplicado en el tiempo de duración de la trayectoria")

    plt.show()

if __name__ == "__main__":
    main()


#
# env = UR5_EnvTest(simulation_frames=1, torque_control= 0.01, distance_threshold=0.05, gui=True)
# ac =  torch.load("/home/alexis/Documentos/repos/TEG/agents_old/ddpg4/pyt_save/model.pt")

# for i_episode in range(50):
#     print('---------------------------------------------------------------')
#     print("estoy en pisodio",i_episode)
#     print('---------------------------------------------------------------')
#
#     observation = env.reset()
#
#     for t in range(500):
#
#         env.render()
#
#         # action = env.action_space.sample() #agente random
#         action = ac.act(torch.as_tensor(observation, dtype=torch.float16)) # agente vpg
#
#         observation, reward, done, info = env.step(action)
#
#         if done:
#             print("la tarea es resuelta en:", t * 0.002)
#             break
#
# env.close()


# doctores


