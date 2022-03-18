import mujoco_py
import torch


agent =  torch.load("/home/alexis/Documentos/repos/TEG/agents/vpg/pyt_save/model.pt")

#creacion del modelo del robot
robot = mujoco_py.load_model_from_path("RobotEnv/assets/UR5/robotModelV2.xml")
sim = mujoco_py.MjSim(robot)
viewer = mujoco_py.MjViewer(sim)

#obtener el tiempo de la simulación
t, _, _, _, _ = sim.get_state()

#obtener observación que usara el agente
def get_observation(self, sim):
    """
        Esta función retorna la posicion y velocidad de las articulaciones y
        la posición xyz de la garra.
    """
    gripper_position = sim.data.get_body_xpos('ee_link')
    joints_position = sim.data.qpos.flat.copy()
    joints_velocity = sim.data.qvel.flat.copy()

    observation = np.concatenate(
        (gripper_position, joints_position, joints_velocity)
    )

    return observation
