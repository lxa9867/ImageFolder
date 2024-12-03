import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return x * self.std + self.mean