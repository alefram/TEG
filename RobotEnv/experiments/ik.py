# calcular los valores de los angulos 

from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik
from dm_control.mujoco.wrapper import mjbindings 
import numpy as np

mjlib = mjbindings.mjlib

# cargar modelos del brazo en string

class _ResetArm:

  def __init__(self, seed=None):
    self._rng = np.random.RandomState(seed)
    self._lower = None
    self._upper = None

  def _cache_bounds(self, physics):
    self._lower, self._upper = physics.named.model.jnt_range[_JOINTS].T
    limited = physics.named.model.jnt_limited[_JOINTS].astype(bool)
    # Positions for hinge joints without limits are sampled between 0 and 2pi
    self._lower[~limited] = 0
    self._upper[~limited] = 2 * np.pi

  def __call__(self, physics):
    if self._lower is None:
      self._cache_bounds(physics)
    # NB: This won't work for joints with > 1 DOF
    new_qpos = self._rng.uniform(self._lower, self._upper)
    physics.named.data.qpos[_JOINTS] = new_qpos

physics = mujoco.Physics.from_xml_path("../assets/UR5/robotModelV3.xml")


# hacer la cinematica inversa

_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
_TOL = 1.2e-14
_MAX_STEPS = 100
_MAX_RESETS = 10
_INPLACE = [False, True]
    # x = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2,
    #     0.15, 0.1, 0.05, 0.0, -0.05, -0.1, -0.15, -0.15, -0.2]
_TARGET = np.array([-0.15, 0.45])
_SITE = "test"

count = 0
physics2 = physics.copy(share_model=True)
resetter = _ResetArm(seed=0)   

while True:
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
    if result.success:
        break
    elif count < _MAX_RESETS:
        resetter(physics2)
        count+=1
    else:
        raise RuntimeError("fallo en encontrar solución dentro de %i" % _MAX_RESETS)

print("angulos posibles:", result.qpos,"\n \n", "error:", result.err_norm)