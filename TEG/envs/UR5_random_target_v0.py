import mujoco
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces

# convertir la observación del ambiente al espacio de observación con sus limites
def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float16)
        high = np.full(observation.shape, float("inf"), dtype=np.float16)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space



class UR5_EnvTest(gym.Env):
    """
    ### argumentos iniciales:

    1. simulation_frames: cantidad de pasos de simulación utilizando una acción del agente
    2. torque_control: constante utilizada en el sistema de recompenza para controlar el torque
    3. distance_threshold: distancia final que indica que la garra llego al objetivo
    4. Gui: Booleano que indica si se permite visualización del brazo manipulador.

    """
    def __init__(self,
                simulation_frames,
                torque_control,
                distance_threshold):

        #inicializar configuraciones de la simulacion
        self.simulation_frames = simulation_frames
        self.C_a = torque_control
        self.distance_threshold = distance_threshold

        #inicializar el modelo del robot
        model_path = "Model-V0.xml"
        fullpath = os.path.join(
            os.path.dirname(__file__), "../assets/UR5", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        #init simulation
        self.robot = mujoco.MjModel.from_xml_path(fullpath)
        self.sim = mujoco.MjData(self.robot)

        #configurar actuadores
        self.init_qpos = [0.2, 1.8, 1.8 ,0.3, 0.7, 0.5]
        self.init_qvel = [0,0,0,0,0,0]
        self.num_actuators = len(self.sim.ctrl)
        self.qpos_bounds = np.array(((-1, 1), (0, 2), (0, 2), (0, 2), (0, 2), (-1, 1)), dtype=object) # rango de articulaciones

        #configurar los espacio de acción
        bounds = self.robot.actuator_ctrlrange.copy().astype(np.float16)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high,dtype=np.float16)

        #configurar espacio observado
        observation = self.get_observation()
        self.observation_space = convert_observation_to_space(observation)

        #configurar el target
        self.target_bounds = np.array(((-0.3, 0.1), (-0.3, 0.3), (0.45, 0.5)), dtype=object) #limites del target a alcanzar


        self.reset()

    def reset(self):

        #inicializar la posición de un objeto eleatorio para iniciar el episodio
        self.reset_target()

        self.sim.qpos[:] = self.init_qpos
        self.sim.qvel[:] = self.init_qvel

        mujoco.mj_forward(self.robot, self.sim)

        return self.get_observation()


    def step(self, action):

        #inicializar variables
        done = False
        reward = 0
        action = np.clip(action, self.action_space.low, self.action_space.high) # me aseguro que no cambiamos la accion fuera de los limites del target

        # aplicar acción de control a la simulación
        self.do_simulation(action,self.simulation_frames)

        # obtendo la observacion es decir el siguiente estado
        observation = self.get_observation()

        # obtengo la recompensa
        reward = self.compute_reward(observation, action)

        # verifico que la garra este al menos de 5cm dando recompensa 1 y terminar el episodio
        # aqui se considera la lograda y terminada
        if (reward == 1):
            done = True

        info = self.get_info(observation)

        return observation, reward, done, info

    ##### funciones utiles ######

    def get_observation(self):
        """
            Esta función retorna la posicion y velocidad de las articulaciones y
            la posición xyz de la garra.
        """
        left_finger = self.sim.body("left_inner_finger").xpos.astype(np.float16)
        right_finger = self.sim.body("right_inner_finger").xpos.astype(np.float16)

        gripper_position = ((left_finger[0] + right_finger[0])/2, (left_finger[1] + right_finger[1])/2, (left_finger[2] + right_finger[2])/2)

        target_position = self.sim.geom("target").xpos
        joints_position = self.sim.qpos.flat.copy().astype(np.float16)
        joints_velocity = self.sim.qvel.flat.copy().astype(np.float16)

        observation = np.concatenate(
            (gripper_position, target_position, joints_position, joints_velocity)
        )

        return observation


    def reset_target(self):
        """
        Esta función resetea para la posición del goal de manera aleatoria.
        """
        # crear una posición del goal aleatorio
        self.goal = np.random.rand(3) * (self.target_bounds[:, 1] -
                                         self.target_bounds[:, 0]
                                         ) + self.target_bounds[:, 0]
        
        self.sim.geom("target").xpos = self.goal

    def do_simulation(self, ctrl, n_frames):
        """
        Esta función permite aplicar control en n cuadros de simulación
        estos pasos son distintos de los pasos del agente
        los simulation frames son los pasos de simulación utilizando una acción de control

        """

        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("dimensión  de las acción no concuerda con el controlador")

        self.sim.ctrl[:] = ctrl

        #este es el frame skip que da la relación con el controlador del brazo a simular
        for _ in range(n_frames):
            mujoco.mj_step(self.robot,self.sim)

    def compute_reward(self, state, action):
        """
        Esta función computa el sistema de recompensa.
        """
        gripper_position = np.array([state[0], state[1], state[2]])
        target_position = self.goal.copy().astype(np.float16)


        distance_norm = np.linalg.norm(target_position - gripper_position).astype(np.float16)

        action_norm = np.linalg.norm(action).astype(np.float16)

        if (distance_norm < self.distance_threshold):
            return 1

        return (-distance_norm - self.C_a * action_norm).astype(np.float16)

    def get_info(self, observation):
        """
        Esta función permite obtener datos utiles.

        ### descripción
        - gripper_posicion: posición xyz del efector final.
        - target position: posición objetivo

        - j_posicion: posición de  las articulaciones
        - j_velocity: velocidad de las articulaciones
        - dist: distancia entre el efector final y el goal

        """
        gripper_position = self.sim.body("ee_link").xpos.astype(np.float16)
        target_position = self.sim.geom("target").xpos.astype(np.float16)

        info = {
            'gripper_position': gripper_position,
            'target_position': target_position,
            'dist': np.linalg.norm(target_position - gripper_position).astype(np.float16),
            'observation': self.get_observation(),
            'j_position': self.sim.qpos.flat.copy().astype(np.float16),
            'j_velocity': self.sim.qvel.flat.copy().astype(np.float16),
        }

        return info
