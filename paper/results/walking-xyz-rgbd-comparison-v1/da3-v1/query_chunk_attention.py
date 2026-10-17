"""Inference-only query chunking; each query still attends to all keys and values."""

from contextlib import contextmanager


@contextmanager
def query_chunk_attention(chunk_size=256):
    import torch
    import torch.nn.functional as functional

    if chunk_size < 1:
        raise ValueError("positive chunk size required")
    original = functional.scaled_dot_product_attention

    def chunked(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if dropout_p != 0 or is_causal:
            raise ValueError("query chunking supports noncausal dropout-free inference only")
        pieces = []
        for start in range(0, query.shape[-2], chunk_size):
            mask = attn_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start : start + chunk_size, :]
            pieces.append(
                original(
                    query[..., start : start + chunk_size, :],
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                    **kwargs,
                )
            )
        return torch.cat(pieces, dim=-2)

    functional.scaled_dot_product_attention = chunked
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original
