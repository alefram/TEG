import mujoco_py
import os


def create_simulation(robot_path):
    """Crea la simulación dado la ruta del modelo de simulación"""
    
    fullpath = os.path.join(
        os.path.dirname(__file__), "../assets/UR5", robot_path)
    if not os.path.exists(fullpath):
        raise IOError("File %s does not exist" % fullpath)

    robot = mujoco_py.load_model_from_path(fullpath)
    simulation = mujoco_py.MjSim(robot)

    return simulation

def create_viewer(simulation):
    """Crea el viewer para visualizar la simulación"""
    viewer = mujoco_py.MjViewer(simulation)

    
# #TODO: terminar cinematica inversa para obtener posiciones de los joints
# def inverse_kinematics(self, target_pos=None, target_quad=None, tol=1e-6, stop_steps=100):
#     """cinematica inversa"""

#     success = False

#     # paso 1 obtener posicion y la matriz de rotacion de el link base
#     base_pos = self.sim.data.get_body_xpos("base_link")        
#     base_quad = self.sim.data.get_body_xquat("base_link")

#     # paso 2 obtener la posicion del target_link
#     target_pos = self.sim.data.get_body_xpos("wrist_3_link")
#     target_quad = self.sim.data.get_body_xquat("wrist_3_link")

#     #paso 3 definir el vector q que tiene los angulos desde la base hasta el target
#     current_q = self.sim.data.qpos.flat.copy()


#     for steps in range(stop_steps):
        
#         #paso 4 calculate with foward kinematics with mujoco
        
    

#         #step 5 calculate error in position and attitude
#         if target is not None:
#             delta_pos = target_pos - current_target_pos
#             delta_pos_norm += np.linalg.norm(delta_pos)
        
#         if target_quad is not None:
#             #TODO: hacer error rotacional
#             delta_quad = 0k

        
#         #paso 6 si es tolerable entonces terminar
#         if delta_pos < tol:
#             print("Convergio")
#             success = True
#             break
#         #paso 7 si no es tolerable entonces calcular delta_q 
#         else:
#             #TODO: calcular el delta_q
#             break

        
        
#         #paso 8 actualizar los angulos de los joints como q = q + delta_q
#         current_q = current_q + delta_q

#     return current_q, success
