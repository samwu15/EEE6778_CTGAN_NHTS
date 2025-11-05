
import torch.nn as nn

class TinyGen(nn.Module):
    """Tiny generator for forward test."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3), nn.ReLU(True),
            nn.Conv2d(64, 3, 7, 1, 3), nn.Tanh()
        )
    def forward(self, x): return self.net(x)