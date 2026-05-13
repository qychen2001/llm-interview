import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        """
        q: (batch, num_heads, seq_len, head_dim)
        k, v: (batch, num_kv_heads, seq_len, head_dim)   # MQA 时 num_kv_heads=1
        mask: (batch, 1, seq_len, seq_len) 或 (batch, 1, 1, seq_len)
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int = 1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        assert num_kv_heads == 1 or num_heads % num_kv_heads == 0, (
            "num_heads 必须能被 num_kv_heads 整除"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads  # 新增
        self.head_dim = hidden_dim // num_heads

        # ==================== QKV 投影 ====================
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)  # Q 仍然是 full heads

        # K、V 只投影到 num_kv_heads 个 head
        kv_dim = num_kv_heads * self.head_dim
        self.k_linear = nn.Linear(hidden_dim, kv_dim)
        self.v_linear = nn.Linear(hidden_dim, kv_dim)

        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, mask=None):
        batch_size, seq_len, _ = x.shape

        # 1. 计算 Q、K、V
        q = self.q_linear(x)  # (B, S, hidden_dim)
        k = self.k_linear(x)  # (B, S, num_kv_heads * head_dim)
        v = self.v_linear(x)

        # 2. Reshape Q → (B, num_heads, S, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        # 3. Reshape K、V → (B, num_kv_heads, S, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. 【关键】把 KV repeat 到和 Q 相同的 head 数量
        if self.num_kv_heads != self.num_heads:
            # repeat: (B, num_kv_heads, S, D) -> (B, num_heads, S, D)
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # 5. Attention
        attn_out, _ = self.attention(q, k, v, mask=mask)

        # 6. 合并 heads
        attn_out = attn_out.transpose(1, 2)  # (B, S, num_heads, head_dim)
        attn_out = attn_out.contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 7. 输出投影
        output = self.out_linear(attn_out)
        return output
