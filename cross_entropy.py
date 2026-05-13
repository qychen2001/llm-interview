import torch


def cross_entropy_loss(logits, targets):
    """
    手写交叉熵 Loss
    logits: [batch, vocab]
    targets: [batch] 类别索引
    """
    batch_size = logits.shape[0]

    # 1. 数值稳定的 LogSoftmax
    # log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    log_sum_exp = max_logits + torch.log(
        torch.sum(torch.exp(logits - max_logits), dim=1, keepdim=True)
    )
    log_probs = logits - log_sum_exp

    # 2. NLL Loss (Negative Log Likelihood)
    # 取出目标类别对应的 log 概率
    # targets 需要 unsqueeze 才能 gather
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # 3. 取平均
    loss = -torch.mean(target_log_probs)
    return loss
