"""Eager grouped-query attention with fp32 score arithmetic to avoid fp16 overflow."""

import torch
from torch.nn import functional as F


def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights
