import torch
import torch.nn as nn


class SiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # TODO: create 2 linear layers
        self.W = nn.Linear(in_features, out_features)  # main linear layer
        self.W_g = nn.Linear(in_features, out_features)  # gated linear layer
  
    def forward(self, x):
        # TODO: implement SiGLU W * x * sigma (W_g * x)
        gated_output = self.W(x) * torch.sigmoid(self.W_g(x))
        return gated_output
