import mujoco_py
import os
import numpy as np

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

def random_target(target_bounds, geom_pos, sim):
    """generar una posición aleatoria del target"""

    goal = np.random.rand(3) * (target_bounds[:, 1] -
                                 target_bounds[:, 0]
                                 ) + target_bounds[:, 0]

    geom_positions = sim.model.geom_pos.copy()
    prev_goal_location = geom_positions[geom_pos]

    geom_positions[geom_pos] = goal
    sim.model.geom_pos[:] = geom_positions

    return goal

def post_target(sim, target, geom_pos):
    simulation_positions = sim.model.geom_pos.copy()
    simulation_positions[geom_pos] = target
    sim.model.geom_pos[:] = simulation_positions

    return target
