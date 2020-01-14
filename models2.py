import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Generator, self).__init__()
        self.input_shape = input_shape

        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, 0.8)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512, 0.8)
        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024, 0.8)
        self.fc5 = nn.Linear(1024, int(np.prod(self.input_shape)))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.fc4(x)), 0.2)
        x = self.fc5(x)
        x = torch.tanh(x)
        x = x.view(x.size(0), *self.input_shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape

        self.fc1 = nn.Linear(int(np.prod(self.input_shape)), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
