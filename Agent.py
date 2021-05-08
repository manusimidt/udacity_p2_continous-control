import numpy as np
import random
from collections import namedtuple, deque

from typing import Tuple

from networks import ActorNetwork, CriticNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

"""
For some reason the pytorch function .to(device) takes forever to execute.
Probably some issue between my cuda version and the pytorch version (0.4.0) that unityagents requires
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
|  0%   59C    P0    28W / 130W |    679MiB /  5941MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
"""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


class Agent:
    """ This class represents the reinforcement learning agent """

    def __init__(self, state_size: int, action_size: int, hidden_sizes: [int] = (64,),
                 gamma: float = 0.99, lr: float = 0.001, tau: float = 0.001,
                 buffer_size: int = 100000, batch_size: int = 64, update_rate: int = 5,
                 seed: int = int(random.random() * 100)):
        self.tau = tau

        self.actor_local = ActorNetwork(state_size, action_size).to(device)
        self.actor_target = ActorNetwork(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr)

        self.critic_local = CriticNetwork(state_size, action_size).to(device)
        self.critic_target = CriticNetwork(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr)

        pass

    def step(self, state, action, reward, next_state):
        pass

    def act(self, state):
        pass

    def learn(self, experiences, gamma):
        pass

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - self.tau) * target_param.data)


class ReplayBuffer:
    """ FiFo buffer storing experience tuples of the agent """

    def __init__(self, action_size: int, buffer_size: int, batch_size: int):
        """
        Initialize Buffer
        :param action_size: dimension of each action
        :param buffer_size: maximum amount of experiences the buffer saves
        :param batch_size: size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def reset(self):
        self.memory.clear()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
