import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        """
        q, k, v 形状：(batch_size, num_heads, seq_len, head_dim)
        mask 形状：(batch_size, 1, 1, seq_len) 或 (batch_size, 1, seq_len, seq_len)
        """
        # 1. 计算 Q 和 K 的点积 (Batch, Heads, Seq, Seq)
        # q: (Batch, Heads, Seq, Head_Dim)
        # k: (Batch, Heads, Seq, Head_Dim) -> transpose(-2, -1) 变为 (Batch, Heads, Head_Dim, Seq)
        # 目的：将 K 的最后两个维度互换，使 Q 的最后一维与 K 的倒数第二维匹配，从而进行矩阵乘法计算 Attention Score
        scores = torch.matmul(q, k.transpose(-2, -1))

        # 2. 缩放 (Scale)
        # q.size(-1) 是 head_dim。缩放是为了防止点积结果过大导致 Softmax 梯度消失
        scale = math.sqrt(q.size(-1))
        scores = scores / scale

        # 3. 应用 Mask (如果是解码器，需要遮住未来的 token)
        if mask is not None:
            # mask 中为 0 的地方保留，为 1 的地方遮住 (通常 mask 是 1 表示遮住)
            # 这里假设 mask 是 1 表示需要遮住的位置，我们将其设为负无穷
            scores = scores.masked_fill(mask == 1, -1e9)

        # 4. Softmax 归一化
        # 对最后一个维度 (key 的 seq_len) 进行归一化，得到每个 token 对其他 token 的注意力权重
        # attn_weights 形状: (Batch, Heads, Seq, Seq)
        attn_weights = torch.softmax(scores, dim=-1)

        # 5. 加权求和 V
        # (Batch, Heads, Seq, Seq) @ (Batch, Heads, Seq, Head_Dim) -> (Batch, Heads, Seq, Head_Dim)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 1. QKV 的线性投影层
        # 输入 (B, S, H) -> 输出 (B, S, H)
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        # 2. 最后的输出投影层
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.attention = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        """
        x 形状：(batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        # 1. 计算 Q, K, V (B, S, H)

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # 2. 拆分多头 (Reshape & Transpose)
        # 目标形状：(B, num_heads, S, head_dim)
        # 先变成 (B, S, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 再转置成 (B, num_heads, S, head_dim)，方便后面做矩阵乘法
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 3. 进入注意力计算

        # attn_out 形状：(B, num_heads, S, head_dim)
        attn_out, _ = self.attention(q, k, v, mask=mask)

        # 4. 合并多头 (Concat & Project)
        # 先转置回 (B, S, num_heads, head_dim)
        attn_out = attn_out.transpose(1, 2)
        # 再 reshape 成 (B, S, hidden_dim)
        attn_out = attn_out.contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 5. 最后过一层线性层
        output = self.out_linear(attn_out)

        return output


if __name__ == "__main__":
    # 模拟输入：Batch=2, Seq_Len=5, Hidden_Dim=64
    x = torch.randn(2, 5, 64)

    # 创建一个简单的因果 Mask (下三角为 0，上三角为 1)
    # 表示当前 token 只能看前面的，不能看后面的
    mask = torch.triu(torch.ones(5, 5), diagonal=1).unsqueeze(0).unsqueeze(0)

    mha = MultiHeadAttention(hidden_dim=64, num_heads=8)
    out = mha(x, mask=mask)

    print(f"输入形状：{x.shape}")
    print(f"输出形状：{out.shape}")
    # 预期输出：torch.Size([2, 5, 64])
