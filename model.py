import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(28 * 28, 512), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 10), nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
