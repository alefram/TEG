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
    return viewer


