import numpy as np
import torch
import  torch.nn as nn
from torch.optim import Adam
import gym
from torch.distributions.normal import Normal
from RobotEnv.envs.UR5_Env import UR5_EnvTest


#perceptron multicapa MPL
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        action = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), action()]
    
    return nn.Sequential(*layers)

# combinar dimensiones
def combined_shape(length, shape=None):
    if shape in None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

# politica de control
class Policy(nn.Module):
    """ implementación de la politica inicial para acciones continuas"""
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def distribution(self, obs):
        """
        distribución gaussiana diagonal
        """
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob_from_distribution(self, pi, action):
        return pi.loq_prob(action).sum(axis=-1) # explicar despues

    def foward(self, obs,  action=None):
        """
        produce una distribución de acciones para una observación dada y 
        opcionalmente computa el log likelihood para una acción dada bajo esas distribuciónes
        """
        pi = self.distribution(obs)
        logp_a = None
        if action is not None:
            logp_a = self.log_prob_from_distribution(pi, action)

        return pi, logp_a

# colección de trayectorias D_k = {tau_i} tambien conocido como buffer
class memory:
    """
    Esta es la memoria del agente que almacena las trayectorias experimentadas con el ambiente
    usando Generalized Advantage Estimation para calcular los advantages de un par acción estado
    """
    def __init__(self,  obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_memory = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_memory = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.advantage_memory = np.zeros(size, dtype=np.float32)
        self.reward_memory = np.zeros(size, dtype=np.float32)
        self.j_memory = np.zeros(size, dtype=np.float32)
        self.v_memory = np.zeros(size, dtype=np.float32)
        self.logp_memory = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store_data(self, obs, act, reward, v, logp):
        """
        agrega un paso de interacción con el ambiente a la memoria
        """
        assert self.ptr < self.max_size
        self.obs_memory[self.ptr] = obs
        self.act_memory[self.ptr] = act
        self.reward_memory[self.ptr] = rew
        self.v_memory[self.ptr] = v
        self.logp_memory[self.ptr] = logp
        self.ptr += 1
    
    # TODO: terminar y quitar el uso de MPI
    def get_data(self):
        """
        esta función te devuelve toda la data de la memoria.
        se llama al  final  del episodio de entrenamiento
        ya se encuentra con los advantages normalizados
        """
        # chequear que la memoria esta full
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # implemento el advantage normalization
        advantage_memory = np.array(self.advantage_memory, dtype=np.float32) #x
        global_sum,  global_n = 
        advantge_mean = global_sum / global_n

        global_sum_sq = 
        advantage_std = np.sqrt(global_sum_sq / global_n) #computar global std

        self.advantage_memory = (self.advantage_memory - advantage_mean) / advantage_std
        data = dict(obs=self.obs_memory, action=self.act_memory, j=self.j_memory,
                    advantage=self.advantage_memory, logp=self.logp_memory)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}




# función valor
class Value_function(nn.Module):
    
    def __init__(self, obs_dim,hidden_sizes, activation):
        super().__init__()
        self.value_network = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.value_network(obs), -1) # para asegurar que v tiene la forma correct

if __name__ == "__main__":

    env = UR5_EnvTest(simulation_frames=10, torque_control= 0.01, distance_threshold=0.5, Gui=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]


    pi = Policy(obs_dim, act_dim, (64,64), activation=nn.Tanh)
    v = Value_function(obs_dim, hidden_sizes=(64,64), activation=nn.Tanh)
    
    print(pi)
    print(v)
