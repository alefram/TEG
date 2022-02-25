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


# función valor
class Value_function(nn.Module):
    
    def __init__(self, obs_dim,hidden_sizes, activation):
        super().__init__()
        self.value_network = mpl([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.value_network(obs), -1) # para asegurar que v tiene la forma correct

if __name__ == "__main__":

    env = UR5_EnvTest(simulation_frames=10, torque_control= 0.01, distance_threshold=0.5, Gui=False)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape


    policy = Policy(obs_dim, act_dim, (64,64), activation=nn.Tanh)
    print(policy)
