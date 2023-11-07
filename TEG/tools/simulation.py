from mujoco import viewer

def render_simulation(loader):
    """Create viewer to render simulation"""
    
    viewer.launch(loader=loader)

