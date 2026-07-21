import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dropout(x)
        out = self.linear(x)
        out = self.activation(out)
        out = out + residual
        return out