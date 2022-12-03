"""experimento 1 """

from RobotEnv.envs.UR5_Env import UR5_EnvTest
from RobotEnv.tools import simulation
from RobotEnv.tools import controllers
import numpy as np
import os
import argparse
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from dm_control import mujoco
from RobotEnv.tools.ik import inverse_kinematics

# construir los inputs del usuario

parser = argparse.ArgumentParser()

parser.add_argument("--agent1", help="agente 1")
parser.add_argument("--agent2", help="agente 2")
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
agent = args.agent1
agent2 = args.agent2
PHYSICS_PATH = "../RobotEnv/assets/UR5/robotModelV3.xml"

sim = simulation.create_simulation("robotModelV3.xml")


# loop de recorido de la trayectoria
def main():
    win = 0
    win2 = 0
    average_time = []
    average_time2 = []
    targets = []

    controller = controllers.Manipulator_Agent(agent, sim, render=render)
    
    print("AGENTE 1")
    for i in range(episodes):
        
        print('---------------------')
        print("episodio", i)
        print('---------------------')

        target = simulation.random_target(target_bounds, geom_pos, sim)
        controller.reset()

        position, qpos, control, done, steps = controller.move_to(np.array(target), 
                                           distance_threshold=dist, timer=timer)
        
        average_time.append(steps)
        targets.append(target)

        if (done):
            win += 1

    controller.change_agent(agent2)
    print("AGENTE 2")
    for j in range(episodes):
    
        print('---------------------')
        print("episodio", j)
        print('---------------------')

        target = simulation.post_target(sim, targets[j], geom_pos)
        controller.reset()

        position2, qpos2, control2, done2, steps2 = controller.move_to(np.array(target), 
                                           distance_threshold=dist, timer=timer)
        
        average_time2.append(steps2)

        if (done2):
            win2 += 1
    
# ------------------------------RESULTADOS--------------------------------------

# variables de ayuda

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
    steps_array2 = [i for i in range(steps2)]
    
    targetx = [np.array(target[0]) for i in range(steps)]
    targety = [np.array(target[1]) for i in range(steps)]
    targetz = [np.array(target[2]) for i in range(steps)]

    targetx2 = [np.array(target[0]) for i in range(steps2)]
    targety2 = [np.array(target[1]) for i in range(steps2)]
    targetz2 = [np.array(target[2]) for i in range(steps2)]

    print("-------------------------")
    print("AGENTE 1")
    print("TASA DE ACIETOS")
    print(win, "de", episodes)
    print("PROMEDIO DE TIEMPO")
    print((sum(average_time) / len(average_time))*0.002)
    print("-------------------------")

    print("-------------------------")
    print("AGENTE 2")
    print("TASA DE ACIETOS")
    print(win2, "de", episodes)
    print("PROMEDIO DE TIEMPO")
    print((sum(average_time2) / len(average_time2))*0.002)
    print("-------------------------")

# # TODO:hacer grafica de tasas de aciertos

# # grafica de trayectoria final.

#     fig1 = plt.figure()
#     ax1 = fig1.add_subplot(projection="3d")
#     # fig1.suptitle("Trayectoria para un objetivo")
#     ax1.plot(position["pos_x"], position["pos_y"], position["pos_z"],
#             linewidth=2.0, label="agente 1")
#     ax1.plot(position2["pos_x"], position2["pos_y"], position2["pos_z"],
#             linewidth=2.0, label="agente 2")
#     ax1.plot(targetx, targety, targetz, 'o', 
#         linewidth=2.0, label="objetivo")
#     ax1.set_xlabel("posición x")
#     ax1.set_ylabel("posición y")
#     ax1.set_zlabel("posición z")
#     ax1.grid(True)
#     ax1.legend()


#     fig2, ax2 = plt.subplots(1,1)
#     # fig2.suptitle("Trayectoria del efector final en x por iteración")
#     sup = targetx + np.array([dist for i in range(steps)])
#     inf = targetx - np.array([dist for i in range(steps)])
#     ax2.plot(steps_array, position['pos_x'], linewidth=2.0, label="agente 1")
#     ax2.plot(steps_array2, position2['pos_x'], linewidth=2.0, label="agente 2")

#     ax2.plot(steps_array, targetx, linestyle='--', linewidth=2.0, 
#             label="objetivo")
#     ax2.fill_between(steps_array, sup, inf, alpha=0.2)
#     ax2.set_xlabel("iteraciones")
#     ax2.set_ylabel("x(s)")
#     ax2.grid(True)
#     ax2.legend()


#     fig3, ax3 = plt.subplots(1,1)
#     # fig3.suptitle("Trayectoria del efector final en y por iteración")
#     sup2 = targety + np.array([dist for i in range(steps)])
#     inf2 = targety - np.array([dist for i in range(steps)])
#     ax3.plot(steps_array, position['pos_y'], linewidth=2.0, label="agente 1")
#     ax3.plot(steps_array2, position2['pos_y'], linewidth=2.0, label="agente 2")

#     ax3.plot(steps_array, targety, linestyle='--', linewidth=2.0, 
#             label="objetivo")
#     ax3.fill_between(steps_array, sup2, inf2, alpha=0.2, label="umbral objetivo")
#     ax3.set_xlabel("iteraciones")
#     ax3.set_ylabel("y(s)")
#     ax3.grid(True)
#     ax3.legend()

#     fig4, ax4 = plt.subplots(1,1)
#     # fig4.suptitle("Trayectoria del efector final en z por iteración")
#     sup3 = targetz + np.array([dist for i in range(steps)])
#     inf3 = targetz - np.array([dist for i in range(steps)])
#     ax4.plot(steps_array, position['pos_z'], linewidth=2.0, label="agente 1")
#     ax4.plot(steps_array2, position2['pos_z'], linewidth=2.0, label="agente 2")

#     ax4.plot(steps_array, targetz, linestyle='--', linewidth=2.0, 
#             label="objetivo")
#     ax4.fill_between(steps_array, sup3, inf3, alpha=0.2, label="umbral objetivo")
#     ax4.set_xlabel("iteraciones")
#     ax4.set_ylabel("z(s)")
#     ax4.grid(True)
#     ax4.legend()


# # graficar error de trayectoria cartesiana
#     error_x = np.subtract(np.array(targetx), np.array(position["pos_x"]))
#     error_y = np.subtract(np.array(targety), np.array(position["pos_y"]))
#     error_z = np.subtract(np.array(targetz), np.array(position["pos_z"]))
#     error_x2 = np.subtract(np.array(targetx2), np.array(position2["pos_x"]))
#     error_y2 = np.subtract(np.array(targety2), np.array(position2["pos_y"]))
#     error_z2 = np.subtract(np.array(targetz2), np.array(position2["pos_z"]))
#     reference = [ 0 for i in range(steps)]
#     reference2 = [ 0 for i in range(steps2)]

#     fig5, ax5 = plt.subplots(1,1)
#     ax5.plot(steps_array, error_x, linewidth=2.0, label="error en eje x")
#     ax5.plot(steps_array, error_y, linewidth=2.0, label="error en eje y")
#     ax5.plot(steps_array, error_z, linewidth=2.0, label="error en eje z")
#     ax5.plot(steps_array, reference, linewidth=2.0, label="referencia", 
#             linestyle='--')
#     ax5.set_xlabel("Iteraciones")
#     ax5.set_ylabel("Error del agente 1")
#     ax5.grid(True)
#     ax5.legend()

#     fig52, ax52 = plt.subplots(1,1)
#     ax52.plot(steps_array2, error_x2, linewidth=2.0, label="error en eje x")
#     ax52.plot(steps_array2, error_y2, linewidth=2.0, label="error en eje y")
#     ax52.plot(steps_array2, error_z2, linewidth=2.0, label="error en eje z")
#     ax52.plot(steps_array2, reference2, linewidth=2.0, label="referencia", 
#             linestyle='--')
#     ax52.set_xlabel("Iteraciones")
#     ax52.set_ylabel("Error del agente 2")
#     ax52.grid(True)
#     ax52.legend()


# # Colocar la grafica de los angulos aplicados durante la trayectoria.

#     result = inverse_kinematics(physics, _joints, _target, _site, _tol, _max_steps,
#                                 _max_resets, _inplace)

#     fig6, ax6 = plt.subplots(1, 1)
#     reference = [result.qpos[0] for i in range(steps)]
#     ax6.plot(steps_array, qpos["base_link"], linewidth=2.0, label="agente 1")
#     ax6.plot(steps_array2, qpos2["base_link"], linewidth=2.0, label="agente 2")
#     ax6.plot(steps_array, reference, linewidth=2.0, label="objetivo",
#                   linestyle="--", color="gray")
#     ax6.set_xlabel("iteraciones")
#     ax6.set_ylabel("angulo(rad) de base_link")
#     ax6.legend()
#     ax6.grid(True)



#     fig7, ax7 = plt.subplots(1, 1)
#     reference = [result.qpos[1] for i in range(steps)]
#     # ax7.set_title("Angulo aplicado por el actuador shoulder_link")
#     ax7.plot(steps_array, qpos["shoulder_link"], linewidth=2.0, label="agente 1")
#     ax7.plot(steps_array2, qpos2["shoulder_link"], linewidth=2.0, label="agente 2")

#     ax7.plot(steps_array, reference, linewidth=2.0, label="objetivo",
#                   linestyle="--", color="gray")
#     ax7.set_xlabel("iteraciones")
#     ax7.set_ylabel("angulo(rad) de shoulder_link")
#     ax7.legend()
#     ax7.grid(True)


#     fig8, ax8 = plt.subplots(1, 1)
#     reference = [result.qpos[2] for i in range(steps)]
#     # ax8.set_title("Angulo aplicado por el actuador elbow_link")
#     ax8.plot(steps_array, qpos["elbow_link"], linewidth=2.0, label="agente 1")
#     ax8.plot(steps_array2, qpos2["elbow_link"], linewidth=2.0, label="agente 2")

#     ax8.plot(steps_array, reference, linewidth=2.0, label="objetivo",
#                   linestyle="--", color="gray")
#     ax8.set_xlabel("iteraciones")
#     ax8.set_ylabel("angulo(rad) de elbow_link")
#     ax8.legend()
#     ax8.grid(True)


#     fig9, ax9 = plt.subplots(1, 1)
#     reference = [result.qpos[3] for i in range(steps)]
#     # ax9.set_title("Angulo aplicado por el actuador wrist_1_link")
#     ax9.plot(steps_array, qpos["wrist_1_link"], linewidth=2.0, label="agente 1")
#     ax9.plot(steps_array2, qpos2["wrist_1_link"], linewidth=2.0, label="agente 2")

#     ax9.plot(steps_array, reference, linewidth=2.0, label="objetivo",
#                   linestyle="--", color="gray")
#     ax9.set_xlabel("iteraciones")
#     ax9.set_ylabel("angulo(rad) de wrist_1_link")
#     ax9.legend()
#     ax9.grid(True)

  

#     fig10, ax10 = plt.subplots(1, 1)
#     reference = [result.qpos[4] for i in range(steps)]
#     # ax10.set_title("Angulo aplicado por el actuador wrist_2_link")
#     ax10.plot(steps_array, qpos["wrist_2_link"], linewidth=2.0, label="agente 1")
#     ax10.plot(steps_array2, qpos2["wrist_2_link"], linewidth=2.0, label="agente 2")

#     ax10.plot(steps_array, reference, linewidth=2.0, label="objetivo",
#                   linestyle="--", color="gray")
#     ax10.set_xlabel("iteraciones")
#     ax10.set_ylabel("angulo(rad) de wrist_2_link")
#     ax10.legend()
#     ax10.grid(True)


#     fig11, ax11 = plt.subplots(1, 1)
#     reference = [result.qpos[5] for i in range(steps)]
#     # ax11.set_title("Angulo aplicado por el actuador wrist_3_link")
#     ax11.plot(steps_array, qpos["wrist_3_link"], linewidth=2.0, label="agente 1")
#     ax11.plot(steps_array2, qpos2["wrist_3_link"], linewidth=2.0, label="agente 2")

#     ax11.plot(steps_array, reference, linewidth=2.0, label="objetivo",
#                   linestyle="--", color="gray")
#     ax11.set_xlabel("iteraciones")
#     ax11.set_ylabel("angulo(rad) de wrist_3_link")
#     ax11.legend()
#     ax11.grid(True)



# # error de los angulos
#     reference = [ 0 for i in range(steps)]
#     result1 = [result.qpos[0] for i in range(steps)]
#     result2 = [result.qpos[1] for i in range(steps)]
#     result3 = [result.qpos[2] for i in range(steps)]
#     result4 = [result.qpos[3] for i in range(steps)]
#     result5 = [result.qpos[4] for i in range(steps)]
#     result6 = [result.qpos[5] for i in range(steps)]

#     error1 = np.subtract(result1, qpos["base_link"])
#     error2 = np.subtract(result2, qpos["shoulder_link"])
#     error3 = np.subtract(result3, qpos["elbow_link"])
#     error4 = np.subtract(result4, qpos["wrist_1_link"])
#     error5 = np.subtract(result5, qpos["wrist_2_link"])
#     error6 = np.subtract(result6, qpos["wrist_3_link"])

#     reference2 = [ 0 for i in range(steps2)]
#     result12 = [result.qpos[0] for i in range(steps2)]
#     result22 = [result.qpos[1] for i in range(steps2)]
#     result32 = [result.qpos[2] for i in range(steps2)]
#     result42 = [result.qpos[3] for i in range(steps2)]
#     result52 = [result.qpos[4] for i in range(steps2)]
#     result62 = [result.qpos[5] for i in range(steps2)]

#     error12 = np.subtract(result12, qpos2["base_link"])
#     error22 = np.subtract(result22, qpos2["shoulder_link"])
#     error32 = np.subtract(result32, qpos2["elbow_link"])
#     error42 = np.subtract(result42, qpos2["wrist_1_link"])
#     error52 = np.subtract(result52, qpos2["wrist_2_link"])
#     error62 = np.subtract(result62, qpos2["wrist_3_link"])

#     fig12, ax12 = plt.subplots(1,1, figsize=(10,6))
#     ax12.plot(steps_array, error1, linewidth=2.0, label="base_link agente 1")
#     ax12.plot(steps_array, error2, linewidth=2.0, label="shoulder_link agente 1")
#     ax12.plot(steps_array, error3, linewidth=2.0, label="elbow_link agente 1")
#     ax12.plot(steps_array, error4, linewidth=2.0, label="wrist_1_link agente 1")
#     ax12.plot(steps_array, error5, linewidth=2.0, label="wrist_2_link agente 1")
#     ax12.plot(steps_array, error6, linewidth=2.0, label="wrist_3_link agente 1")
#     ax12.plot(steps_array, reference, linewidth=2.0, label="referencia", 
#             linestyle='--')
#     ax12.set_xlabel("Iteraciones")
#     ax12.set_ylabel("Error del agente 1")
#     ax12.grid(True)
#     ax12.legend()



#     fig122, ax122 = plt.subplots(1,1, figsize=(10,6))
#     ax122.plot(steps_array2, error12, linewidth=2.0, label="base_link agente 2")
#     ax122.plot(steps_array2, error22, linewidth=2.0, label="shoulder_link agente 2")
#     ax122.plot(steps_array2, error32, linewidth=2.0, label="elbow_link agente 2")
#     ax122.plot(steps_array2, error42, linewidth=2.0, label="wrist_1_link agente 2")
#     ax122.plot(steps_array2, error52, linewidth=2.0, label="wrist_2_link agente 2")
#     ax122.plot(steps_array2, error62, linewidth=2.0, label="wrist_3_link agente 2")
#     ax122.plot(steps_array2, reference2, linewidth=2.0, label="referencia", 
#             linestyle='--')
#     ax122.set_xlabel("Iteraciones")
#     ax122.set_ylabel("Error del agente 2")
#     ax122.grid(True)
#     ax122.legend()


# # TODO:colocar la grafica del torque aplicado durante la trayectoria.

#     figA, axA = plt.subplots()
#     axA.plot(steps_array, control["base_link"], linewidth=2.0, label="agente 1")
#     axA.plot(steps_array2, control2["base_link"], linewidth=2.0, label="agente 2")
#     axA.set_xlabel("Iteraciones")
#     axA.set_ylabel("torque(Nm) de base_link")
#     axA.grid(True)
#     axA.legend()


#     figB, axB = plt.subplots()
    
#     axB.plot(steps_array, control["shoulder_link"], linewidth=2.0, label="agente 1")
#     axB.plot(steps_array2, control2["shoulder_link"], linewidth=2.0, label="agente 2")
#     axB.set_xlabel("Iteraciones")
#     axB.set_ylabel("torque(Nm) de shoulder_link")
#     axB.grid(True)
#     axB.legend()


#     figC, axC = plt.subplots()

#     axC.plot(steps_array, control["elbow_link"], linewidth=2.0, label="agente 1")
#     axC.plot(steps_array2, control2["elbow_link"], linewidth=2.0, label="agente 2")
#     axC.set_xlabel("Iteraciones")
#     axC.set_ylabel("torque(Nm) de elbow_link")
#     axC.grid(True)
#     axC.legend()


#     figD, axD = plt.subplots()

#     axD.plot(steps_array, control["wrist_1_link"], linewidth=2.0, label="agente 1")
#     axD.plot(steps_array2, control2["wrist_1_link"], linewidth=2.0, label="agente 2")
#     axD.set_xlabel("Iteraciones")
#     axD.set_ylabel("torque(Nm) de wrist_1_link")
#     axD.grid(True)
#     axD.legend()


#     figE, axE = plt.subplots()

#     axE.plot(steps_array, control["wrist_2_link"], linewidth=2.0, label="agente 1")
#     axE.plot(steps_array2, control2["wrist_2_link"], linewidth=2.0, label="agente 2")
#     axE.set_xlabel("Iteraciones")
#     axE.set_ylabel("torque(Nm) de wrist_2_link")
#     axE.grid(True)
#     axE.legend()



#     figF, axF = plt.subplots()

#     axF.plot(steps_array, control["wrist_3_link"], linewidth=2.0, label="agente 1")
#     axF.plot(steps_array2, control2["wrist_3_link"], linewidth=2.0, label="agente 2")
#     axF.set_xlabel("Iteraciones")
#     axF.set_ylabel("torque(Nm) de wrist_3_link")
#     axF.grid(True)
#     axF.legend()

# # guardar
#     fig1.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-trayectoria3D.png")
#     fig2.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-trayectoriaX.png")
#     fig3.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-trayectoriaY.png")
#     fig4.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-trayectoriaZ.png")
#     fig5.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-errorTrayectoria1.png")
#     fig52.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-errorTrayectoria2.png")
#     fig6.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-anguloBase.png")
#     fig7.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-anguloShoulder.png")
#     fig8.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-anguloElbow.png")
#     fig9.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-anguloWrist1.png")
#     fig10.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-anguloWrist2.png")
#     fig11.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-anguloWrist3.png")
#     fig12.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-errorAngulo1.png")
#     fig122.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-errorAngulo2.png")
#     figA.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-torqueBase.png")
#     figB.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-torqueShoulder.png")
#     figC.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-torqueElbow.png")
#     figD.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-torqueWrist1.png")
#     figE.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-torqueWrist2.png")
#     figF.savefig("/home/alexis/Documentos/repos/tesis/images/resultados/E1-torqueWrist3.png")


    plt.show()

if __name__ == "__main__":
    main()

