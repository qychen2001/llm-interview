import torch


def top_p_filtering(
    logits, top_p=0.9, temperature=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    # 按照温度对 logits 修正
    logits = logits / temperature

    # 对 logits 进行排序，从大到小，按照最后一个维度进行排序
    # 假设 logits 形状为 (Batch, Vocab)，要对的是 logits 的最后一个维度（词汇表维度）
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # 计算 softmax 概率
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 只保留概率最高的 token，直到累积概率超过 top_p
    # 所以累计概率超过 top_p 的 token 都会被过滤掉
    sorted_indices_to_remove = cumulative_probs > top_p  # (Batch, Vocab)
    sorted_indices_to_remove[..., :min_tokens_to_keep] = (
        False  # 保留至少 min_tokens_to_keep 个 token
    )
    # 注意：sorted_logits 是排序后的，需要映射回原始位置
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    # “去哪里” (dim/index)，把“谁” (src) 放进去。
    logits[indices_to_remove] = filter_value
    # 将被移除的 logits 设为 -inf（softmax 后概率为 0）

    return logits


if __name__ == "__main__":
    # 测试 top_p_filtering 函数
    logits = torch.tensor([[0.1, 0.5, 0.3, 0.2]])
    top_p = 0.7
    filtered_logits = top_p_filtering(logits, top_p=top_p)
    print("Original logits:", logits)
    print(f"Filtered logits (top p={top_p}):", filtered_logits)
