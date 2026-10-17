"""Compare first-layer attention arithmetic without changing its returned output.

Run as a standalone script beside run_vlm_dnf_pilot.py on the research host.
The original eager attention remains in effect during this numerical audit.
"""

import argparse
import json
from pathlib import Path


def main(checkpoint, image, output, trace):
    if output.exists() or trace.exists():
        raise ValueError("preserve previous attempts")
    import run_vlm_dnf_pilot as runner
    import torch
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as implementation

    original = implementation.eager_attention_forward
    recorded = False

    def stats(tensor):
        finite = torch.isfinite(tensor)
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "nonfinite": int((~finite).sum().item()),
            "maximum_finite_absolute": float(tensor[finite].abs().max().item())
            if finite.any().item()
            else None,
        }

    def audited(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        nonlocal recorded
        if not recorded and type(module).__name__ == "Qwen2_5_VLAttention":
            recorded = True
            repeated = implementation.repeat_kv(key, module.num_key_value_groups)
            raw16 = torch.matmul(query, repeated.transpose(2, 3))
            raw32 = torch.matmul(query.float(), repeated.float().transpose(2, 3))
            scaled16, scaled32 = raw16 * scaling, raw32 * scaling
            masked16 = scaled16 if attention_mask is None else scaled16 + attention_mask
            masked32 = scaled32 if attention_mask is None else scaled32 + attention_mask.float()
            probabilities16 = torch.softmax(masked16, dim=-1, dtype=torch.float32)
            probabilities32 = torch.softmax(masked32, dim=-1, dtype=torch.float32)
            data = {
                "status": "complete",
                "scope": "arithmetic audit; original attention returned unchanged",
                "script_sha256": runner.digest_file(Path(__file__)),
                "runner_sha256": runner.digest_file(Path(runner.__file__)),
                "implementation_sha256": runner.digest_file(Path(implementation.__file__)),
                "image_sha256": runner.digest_file(image),
                "layer": getattr(module, "layer_idx", None),
                "scaling": scaling,
                "statistics": {
                    name: stats(tensor)
                    for name, tensor in {
                        "query": query,
                        "key": key,
                        "value": value,
                        "qk_fp16": raw16,
                        "qk_fp32": raw32,
                        "scaled_fp16": scaled16,
                        "scaled_fp32": scaled32,
                        "softmax_from_fp16": probabilities16,
                        "softmax_from_fp32": probabilities32,
                    }.items()
                },
            }
            trace.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    implementation.eager_attention_forward = audited
    try:
        runner.run(checkpoint, image, output, "geometry", 768, "nf4", True)
    finally:
        implementation.eager_attention_forward = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "image", "output", "trace"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.image, args.output, args.trace)
