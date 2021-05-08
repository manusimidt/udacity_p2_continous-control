import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer) -> tuple:
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, fc1_size: int = 128, fc2_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=fc1_size)
        self.fc2 = nn.Linear(in_features=fc1_size, out_features=fc2_size)
        self.fc3 = nn.Linear(in_features=fc2_size, out_features=output_size)
        self.reset_weights()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

    def reset_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class CriticNetwork(nn.Module):

    def __init__(self, input_size: int, output_size: int, fc1_size: int = 128, fc2_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=fc1_size)
        self.fc2 = nn.Linear(in_features=fc1_size, out_features=fc2_size)
        self.fc3 = nn.Linear(in_features=fc2_size, out_features=output_size)
        self.reset_weights()
        pass

    def forward(self, state, action):
        # todo: how should i insert the action here
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def reset_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
