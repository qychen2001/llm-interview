import numpy as np


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    预计算旋转角度的 cos 和 sin 值（复数形式常称为 cis，即 cos + i sin）
    dim: 特征维度，必须是偶数
    max_seq_len: 最大序列长度
    theta: base 频率参数，通常为 10000
    返回:
        cos: (max_seq_len, dim//2)
        sin: (max_seq_len, dim//2)
    """
    assert dim % 2 == 0, "维度必须为偶数"
    # 计算每个维度的频率 theta_i
    i = np.arange(0, dim, 2)  # i = 0, 2, 4, ..., dim-2
    freqs = 1.0 / (theta ** (i / dim))  # (dim//2,)

    # 生成位置序列
    t = np.arange(max_seq_len)  # (max_seq_len,)
    # 计算角度 t * freqs (广播)
    angles = np.outer(t, freqs)  # (max_seq_len, dim//2)

    cos = np.cos(angles)  # (max_seq_len, dim//2)
    sin = np.sin(angles)  # (max_seq_len, dim//2)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """
    对输入张量 x 应用旋转位置编码
    x: (seq_len, dim) 或 (batch_size, seq_len, dim)，dim 必须为偶数
    cos, sin: (seq_len, dim//2) 预计算好的值
    返回: 编码后的张量，形状与 x 相同
    """
    # 将 x 按最后一维分成两半，reshape 成 (..., seq_len, dim//2, 2)
    # 这样最后一维的两个元素分别是 x_{2i} 和 x_{2i+1}
    orig_shape = x.shape
    seq_len = x.shape[-2]  # 倒数第二维是序列长度
    d = x.shape[-1]
    assert d % 2 == 0
    # 将 x 重塑为 (..., seq_len, d//2, 2)
    x_reshaped = x.reshape(*orig_shape[:-1], d // 2, 2)

    # 分别取出偶数索引和奇数索引部分（即两个分量）
    x_even = x_reshaped[..., 0]  # (..., seq_len, d//2)
    x_odd = x_reshaped[..., 1]  # (..., seq_len, d//2)

    # 应用旋转公式：
    # x'_{2i} = x_{2i} * cos - x_{2i+1} * sin
    # x'_{2i+1} = x_{2i} * sin + x_{2i+1} * cos
    # 注意 cos, sin 的形状为 (seq_len, d//2)，需要广播到与 x_even 一致
    # 如果 x 有 batch 维度，cos/sin 会自动广播到 batch 维（只要维度对齐）
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # 将两部分堆叠回 (..., seq_len, d//2, 2) 形状
    rotated = np.stack([rotated_even, rotated_odd], axis=-1)  # (..., seq_len, d//2, 2)
    # 重塑回原始形状 (..., seq_len, d)
    return rotated.reshape(*orig_shape)


# ========== 示例运行 ==========
if __name__ == "__main__":
    # 参数
    dim = 8
    seq_len = 4
    batch_size = 2

    # 预计算 cos 和 sin
    cos, sin = precompute_freqs_cis(dim, seq_len)
    print("cos 形状:", cos.shape)  # (4, 4) 因为 dim//2 = 4
    print("sin 形状:", sin.shape)

    # 随机生成 Query 和 Key（假设无 batch 或带 batch）
    # 示例 1: 单序列
    q = np.random.randn(seq_len, dim)
    k = np.random.randn(seq_len, dim)
    q_rot = apply_rotary_emb(q, cos, sin)
    k_rot = apply_rotary_emb(k, cos, sin)
    print("\n单序列 RoPE 后 Q 形状:", q_rot.shape)

    # 示例 2: 带 batch 的序列
    q_batch = np.random.randn(batch_size, seq_len, dim)
    k_batch = np.random.randn(batch_size, seq_len, dim)
    q_batch_rot = apply_rotary_emb(q_batch, cos, sin)
    k_batch_rot = apply_rotary_emb(k_batch, cos, sin)
    print("带 batch 的 RoPE 后 Q 形状:", q_batch_rot.shape)

    # 验证内积是否只与相对位置有关（理论性质，这里简单打印差值）
    # 取位置 1 和 2 的 Q/K 计算内积，并与位置 2 和 3 的比较（未严格验证，仅供演示）
    attn1 = np.dot(q_rot[1], k_rot[2])
    attn2 = np.dot(q_rot[2], k_rot[3])
    print(f"内积 (pos1 Q, pos2 K): {attn1:.4f}")
    print(f"内积 (pos2 Q, pos3 K): {attn2:.4f}")
