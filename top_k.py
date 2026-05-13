import torch


def top_k_filtering(logits, top_k=50, temperature=1.0, filter_value=-float("Inf")):
    # 按照温度对 logits 修正
    logits = logits / temperature

    # 获取 top_k 的索引
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]

    # 计算过程说明：
    # 假设 logits 形状为 (Batch, Vocab)
    # 1. torch.topk(logits, top_k)[0] -> 形状 (Batch, top_k)，包含最大的前 k 个值
    # 2. [..., -1] -> 形状 (Batch)，取这 k 个值中的最后一个（即第 k 名的数值，最小的 threshold）
    # 3. [..., -1, None] -> 形状 (Batch, 1)，None 增加一个维度，用于后续广播比较
    # 4. logits < threshold -> 形状 (Batch, Vocab)，将每个值与 threshold 比较，得到掩码

    logits[indices_to_remove] = filter_value
    return logits


if __name__ == "__main__":
    # 测试 top_k_filtering 函数
    logits = torch.tensor([[0.1, 0.5, 0.3, 0.2]])
    top_k = 2
    filtered_logits = top_k_filtering(logits, top_k=top_k)
    print("Original logits:", logits)
    print(f"Filtered logits (top {top_k}):", filtered_logits)
