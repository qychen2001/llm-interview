import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)
    公式: y = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # x: (batch, seq_len, hidden_size) 或 (batch, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


if __name__ == "__main__":
    batch, seq, hidden = 2, 3, 4
    x = torch.randn(batch, seq, hidden)

    ln = LayerNorm(hidden)
    out_ln = ln(x)

    print("Input shape:", x.shape)
    print("Output shape:", out_ln.shape)
