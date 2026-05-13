import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMS 归一化 (Root Mean Square Layer Normalization)
    公式: y = gamma * x / RMS(x), 其中 RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm


if __name__ == "__main__":
    batch, seq, hidden = 2, 3, 4
    x = torch.randn(batch, seq, hidden)

    rn = RMSNorm(hidden)
    out_ln = rn(x)

    print("Input shape:", x.shape)
    print("Output shape:", out_ln.shape)
