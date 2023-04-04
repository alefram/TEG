import os
from mujoco import viewer


def render_simulation(loader):
    """Crea el viewer para visualizar la simulación"""
    render = viewer.launch(loader=loader)

