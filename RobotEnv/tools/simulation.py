import mujoco_py


def create_simulation(robot_path):
    """Crea la simulación dado la ruta del modelo de simulación"""

    robot = mujoco_py.load_model_from_path("./RobotEnv/assets" + robot_path)
    simulation = mujoco_py.MjSim(robot)

    return simulation