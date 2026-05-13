import torch.nn as nn


def sft_loss(logits, labels, ignore_index=-100):
    """
    计算 SFT 的交叉熵损失，自动处理 shift right 和忽略 padding。
    参数:
        logits: 模型输出的 logits，形状 (batch_size, seq_len, vocab_size)
        labels: 真实 token ids，形状 (batch_size, seq_len)，其中填充部分设为 ignore_index
    返回:
        标量损失
    """
    # shift logits 和 labels
    # logits 的最后一个位置没有对应的下一个 token，所以我们取 logits[:, :-1, :]
    # labels 的第一个位置没有对应的前一个预测，所以我们取 labels[:, 1:]
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    # contiguous() 确保内存连续，避免后续计算中的问题

    loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)
    # ignore_index 参数会让损失函数在计算时忽略掉标签为 ignore_index 的位置
    # 这样就不会因为 padding 的 token 影响损失计算了

    # 维度展平计算说明：
    # 假设 shifted_logits 形状为 (Batch, Seq-1, Vocab)，shifted_labels 形状为 (Batch, Seq-1)
    #
    # 1. shifted_labels.view(-1):
    #    彻底展平为 (Batch * (Seq-1))，将整个 batch 的 token 索引压成一维长向量。
    #
    # 2. shifted_logits.view(-1, shifted_logits.size(-1)):
    #    压扁前两维，保留最后一维 Vocab。结果为 (Batch * (Seq-1), Vocab)。
    #    这样每一行就是一个 token 的预测概率分布，正好能与 label 的每个元素一一对应。

    loss = loss_function(
        shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)
    )
    return loss
