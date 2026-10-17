from types import SimpleNamespace

import torch
from paper.fp32_attention_scores import attention


def test_scores_avoid_overflow_before_scaling():
    q = torch.full((1, 2, 3, 128), 30.0, dtype=torch.float16)
    k = torch.full((1, 1, 3, 128), 30.0, dtype=torch.float16)
    v = torch.ones((1, 1, 3, 128), dtype=torch.float16)
    assert not torch.isfinite(q @ k.transpose(-1, -2)).all()
    out, weights = attention(
        SimpleNamespace(num_key_value_groups=2, training=False), q, k, v, None, 128**-0.5
    )
    assert torch.isfinite(out).all() and torch.isfinite(weights).all()
    torch.testing.assert_close(out, torch.ones_like(out))


def test_grouped_masked_attention_matches_explicit_float_reference():
    torch.manual_seed(91)
    q = torch.randn(1, 4, 5, 8, dtype=torch.float16)
    k = torch.randn(1, 2, 5, 8, dtype=torch.float16)
    v = torch.randn(1, 2, 5, 8, dtype=torch.float16)
    mask = torch.zeros(5, 5).masked_fill(torch.ones(5, 5).triu(1).bool(), float("-inf"))
    out, weights = attention(
        SimpleNamespace(num_key_value_groups=2, training=False), q, k, v, mask, 8**-0.5
    )
    scores = (q.float() @ k.repeat_interleave(2, 1).float().transpose(-1, -2)) * 8**-0.5 + mask
    expected_weights = torch.softmax(scores, -1).half()
    expected = (expected_weights @ v.repeat_interleave(2, 1)).transpose(1, 2)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(out, expected)
    assert torch.count_nonzero(weights[..., 0, 1:]) == 0
