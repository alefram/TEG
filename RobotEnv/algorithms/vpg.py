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
        """ distribución gaussiana diagonal """

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
        self.value_network = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

        def forward(self, obs):
            return torch.squeeze(self.value_network(obs), -1) # para asegurar que v tiene la forma correct

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
        self.return_memory = np.zeros(size, dtype=np.float32)
        self.v_memory = np.zeros(size, dtype=np.float32)
        self.logp_memory = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, reward, v, logp):
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

        # TODO:implemento el advantage normalization


        self.advantage_memory = (self.advantage_memory - advantage_mean) / advantage_std
        data = dict(obs=self.obs_memory, action=self.act_memory, return=self.return_memory,
                    advantage=self.advantage_memory, logp=self.logp_memory)


        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

class Agent:
    """
    Este es el agente inteligente que computara partes del algoritmo
    """
    def __init__(self, memory, pi, v=None):
        self.memory = memory
        self.pi = pi
        self.v = v

    def step(self, objs):
        """
        Esta función devuelve guarda la politica pi, la acción  y la probabilidad logaritmica y la función valor siguiente.
        """
        with torch.no_grad():
            pi = self.pi.distribution(obs)
            action = pi.sample()
            logp_a = self.pi.log_prob_from_distribution(pi, action)
            v = self.v(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def compute_rewards_to_go(self, path_slice, rewards):
        """
        computar la función de recompenza rewards to go
        """
        self.memory.return_memory[path_slice] = scipy.signal.lfilter([1], [1, float(-self.memory.gamma)], rewards[::-1], axis=0)[::-1]


    def compute_advantage(self, path_slice, rewards, values):
        """
        aplica el truco de GAE advantage function
        """
        deltas = rewards[:-1] + self.memory.gamma * values[1:] - values[:-1]
        discount = self.memory.gamma * self.memory.lam
        self.memory.advantage_memory[path_slice] = scipy.signal.lfilter([1], [1, float(-discount)], deltas[::-1], axis=0)[::-1]

    def compute_loss_pi(self,data):
        """
        Computa la función de coste de la politica
        """

        obs, action, advantage, logp_old = data['obs'], data['action'], data['advantage'], data['logp']

        #función de perdida de la Policy
        pi, logp = self.pi(obs, action)
        loss_pi = -(logp * advantage).mean()

        # info extra
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_loss_v(data):
        """
        computa la función de coste de la función valor
        """
        obs, return = data['obs'], data['return']
        return ((self.v(obs) - return)**2).mean()

    def get_trajectories(self, steps_per_episode, obs, env, max_episode_len=100,
                         episode_len=0, episode_return=0):
        """
        Esta función me dara la colección  de  trayectorias de D_k = {tau_i} usando la politica Pi_k
        """

        for t in range(steps_per_episode):
            action, v, logp = self.step(torch.as_tensor(obs, dtype=torch.float32))

            next_obs, reward, done, info = env.step(action)
            episode_return += reward
            episode_len += 1

            #guardar en memoria
            self.memory.store(obs, action, reward, v, logp)

            # la observación siguiente ahora es la actual
            obs = next_obs

            timeout = episode_len == max_episode_len
            terminal = done or timeout
            episode_ended = t==steps_per_episode - 1

            if terminal or episode_ended:
                if episode_ended and not (terminal):
                    print('Cuidado: trayectoria cortada por episodio terminado en %done pasos.'%episode_len, flush=True)

                if timeout or episode_ended:
                    _, v, _ = self.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0

                # TODO: crear finish path que en teoria hace rewards to go y compute advantage A_t
                self.memory.finish_path(v)
                if terminal:
                    #TODO: guardar Episode return y episode len para mostrar

                obs, episode_return, episode_len = env.reset(), 0, 0



if __name__ == "__main__":

    env = UR5_EnvTest(simulation_frames=10, torque_control= 0.01, distance_threshold=0.5, Gui=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]


    pi = Policy(obs_dim, act_dim, (64,64), activation=nn.Tanh)
    v = Value_function(obs_dim, hidden_sizes=(64,64), activation=nn.Tanh)

    print(pi)
    print(v)
