import numpy as np
import random
from collections import namedtuple, deque

import copy

from networks import ActorNetwork, CriticNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

random.seed(0)

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

    def __init__(self, state_size: int, action_size: int,
                 gamma: float = 0.99, lr_actor: float = 0.001, lr_critic: float = 0.003,
                 weight_decay: float = 0.0001, tau: float = 0.001,
                 buffer_size: int = 100000, batch_size: int = 64):
        """
        :param state_size: how many states does the agent get as input (input size of neural networks)
        :param action_size: from how many actions can the agent choose
        :param gamma: discount factor
        :param lr_actor: learning rate of the actor network
        :param lr_critic: learning rate of the critic network
        :param weight_decay:
        :param tau: soft update parameter
        :param buffer_size: size of replay buffer
        :param batch_size: size of learning batch (mini-batch)
        """
        self.tau = tau
        self.gamma = gamma

        self.batch_size = batch_size

        self.actor_local = ActorNetwork(state_size, action_size).to(device)
        self.actor_target = ActorNetwork(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        print(self.actor_local)

        self.critic_local = CriticNetwork(state_size, action_size).to(device)
        self.critic_target = CriticNetwork(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        print(self.critic_local)

        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)
        # this would probably also work with Gaussian noise instead of Ornstein-Uhlenbeck process
        self.noise = OUNoise(action_size)

    def step(self, experience: tuple):
        """
        :param experience: tuple consisting of (state, action, reward, next_state, done)
        :return:
        """
        self.memory.add(*experience)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, add_noise: bool = True):
        """ Actor uses the policy to act given a state """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local.forward(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences):
        # Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        # the actor_target returns the next action, this next action is then used (with the state) to estimate
        # the Q-value with the critic_target network

        states, actions, rewards, next_states, dones = experiences

        # region Update Critic
        actions_next = self.actor_target.forward(next_states)
        q_expected = self.critic_local.forward(states, actions)
        q_targets_next = self.critic_target.forward(next_states, actions_next)

        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # minimize the loss
        critic_loss = F.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        # endregion Update Critic

        # region Update actor
        # Compute actor loss
        actions_predictions = self.actor_local.forward(states)
        actor_loss = -self.critic_local.forward(states, actions_predictions).mean()
        # Minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # endregion Update actor

        # region update target network
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        # endregion update target network

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self, local_model, target_model):
        """Copy the weights and biases from the local to the target network"""
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(param.data)

    def reset(self):
        self.noise.reset()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# Copied from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
