from mujoco import viewer

def render_simulation(loader):
    """Create viewer to render simulation"""
    render = viewer.launch(loader=loader)

