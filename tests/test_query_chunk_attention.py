import pytest
import torch
import torch.nn.functional as functional
from paper.query_chunk_attention import query_chunk_attention


@pytest.mark.parametrize("mask_kind", ["none", "full", "broadcast", "bool"])
def test_query_chunks_match_full_attention_and_restore(mask_kind):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 17, 8), torch.randn(2, 3, 19, 8), torch.randn(2, 3, 19, 8)
    mask = None
    if mask_kind in {"full", "bool"}:
        mask = torch.randn(17, 19)
    elif mask_kind == "broadcast":
        mask = torch.randn(1, 19)
    if mask_kind == "bool":
        mask = mask > 0
    original = functional.scaled_dot_product_attention
    expected = original(q, k, v, attn_mask=mask)
    with query_chunk_attention(5):
        actual = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(expected, actual, atol=1e-6)
    assert functional.scaled_dot_product_attention is original


def test_unsupported_causal_mode_fails_and_restores():
    original = functional.scaled_dot_product_attention
    q = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError), query_chunk_attention(2):
        functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert functional.scaled_dot_product_attention is original
