import torch


def stable_softmax(logits, dim=-1):
    """
    数值稳定的 Softmax 实现
    """
    # 1. 减去最大值，防止 exp 溢出
    # keepdim=True 保证形状可以广播
    max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)

    # 2. 计算 exp 的和
    sum_exp_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    # 3. 计算概率
    probs = exp_logits / sum_exp_logits
    return probs
