"""Training-only Qwen LoRA feasibility run; no validation/calibration selection."""

import argparse
import hashlib
import importlib.util
import json
import random
import time
import traceback
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_training(corpus, steps, order="class_smoke"):
    manifest = json.loads((corpus / "manifest.json").read_text())
    path = corpus / "train.jsonl"
    if digest(path) != manifest["exports_sha256"]["train"]:
        raise ValueError("training export changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    allowed = {r["id"]: r for r in manifest["records"] if r["partition"] == "train" and r["status"] == "labeled"}
    if {r["id"] for r in rows} != set(allowed) or len(rows) != len(allowed):
        raise ValueError("training membership mismatch")
    for r in rows:
        record = allowed[r["id"]]
        if r["target"] != record["label"] or r["image"] != record["image"] or r["prompt"] != manifest["prompt"]:
            raise ValueError("training content mismatch")
        if digest(corpus / r["image"]) != record["image_sha256"]:
            raise ValueError("training image changed")
    # This bounded feasibility test deliberately includes all three training labels.
    if order == "class_smoke":
        selected = [next(r for r in rows if r["target"] == label) for label in ("close", "far", "occluded")]
        selected += [r for r in rows if r not in selected]
    elif order == "seeded":
        selected = rows.copy()
        random.Random(0).shuffle(selected)
    else:
        raise ValueError("unknown training order")
    if not 1 <= steps <= len(selected):
        raise ValueError("steps must fit the single training pass")
    return selected[:steps]


def mask_prompt(full_ids, prompt_ids):
    import torch
    n = prompt_ids.shape[1]
    if n >= full_ids.shape[1] or not torch.equal(full_ids[:, :n], prompt_ids):
        raise ValueError("assistant target does not extend the exact image-expanded prompt")
    labels = full_ids.clone()
    labels[:, :n] = -100
    return labels


def run(corpus, checkpoint, output, steps, order="class_smoke"):
    if output.exists():
        raise ValueError("preserve prior training runs")
    rows = select_training(corpus, steps, order)
    output.mkdir(parents=True)
    report = {"status": "starting", "scope": "training feasibility only, not generalization or camera evaluation",
              "training_ids": [r["id"] for r in rows], "corpus_manifest_sha256": digest(corpus / "manifest.json"),
              "train_export_sha256": digest(corpus / "train.jsonl"), "script_sha256": digest(Path(__file__)),
              "checkpoint": str(checkpoint), "steps_requested": steps, "steps": [],
              "order": order,
              "policy": "training only; seed 0; one pass maximum; no calibration/validation files read; class_smoke prioritizes one per class; seeded shuffles every training example",
              "configuration": {"rank": 8, "alpha": 16, "dropout": 0., "learning_rate": 1e-4,
                                "target_modules": ["q_proj", "v_proj"], "visual_min_tokens": 576,
                                "visual_max_tokens": 768, "batch_size": 1, "gradient_clip": 1.}}

    def save():
        (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    save()
    try:
        import torch
        import peft
        import transformers
        import bitsandbytes
        from PIL import Image
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as implementation

        torch.manual_seed(0)
        helper = Path(__file__).with_name("fp32_attention_scores.py")
        spec = importlib.util.spec_from_file_location("training_fp32_scores", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = implementation.eager_attention_forward

        def attention(layer, *args, **kwargs):
            if type(layer).__name__ == "Qwen2_5_VLAttention":
                return module.attention(layer, *args, **kwargs)
            return original(layer, *args, **kwargs)

        implementation.eager_attention_forward = attention
        report.update(versions={"torch": torch.__version__, "peft": peft.__version__,
                               "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__},
                      attention_helper_sha256=digest(helper),
                      checkpoint_files={str(p.relative_to(checkpoint)): digest(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()})
        save()
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, min_pixels=576*28*28, max_pixels=768*28*28)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.float16, attn_implementation="eager",
            device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                                               target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n for n, p in trainable):
            raise ValueError("unexpected trainable base parameters")
        report["trainable_parameters"] = sum(p.numel() for n, p in trainable)
        report["trainable_names"] = [n for n, p in trainable]
        optimizer = torch.optim.AdamW([p for n, p in trainable], lr=1e-4)
        model.train()
        report["status"] = "training"
        save()
        for index, row in enumerate(rows):
            started = time.monotonic()
            user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}
            prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
            full = processor.apply_chat_template([user, {"role": "assistant", "content": row["target"]}], tokenize=False)
            with Image.open(corpus / row["image"]) as im:
                rgb = im.convert("RGB")
                inputs = processor(text=[full], images=[rgb], return_tensors="pt")
                prefix = processor(text=[prompt], images=[rgb], return_tensors="pt")
            inputs["labels"] = mask_prompt(inputs["input_ids"], prefix["input_ids"])
            del prefix
            inputs = inputs.to("cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            grads = [p.grad for n, p in trainable if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise FloatingPointError("missing or nonfinite adapter gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_([p for n, p in trainable], 1.)
            if not torch.isfinite(grad_norm) or grad_norm <= 0:
                raise FloatingPointError("invalid adapter gradient norm")
            optimizer.step()
            item = {"step": index+1, "id": row["id"], "target": row["target"], "loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(grad_norm), "gradient_tensors": len(grads),
                    "supervised_tokens": int((inputs["labels"] != -100).sum()),
                    "image_tokens": int((inputs["input_ids"] == model.config.image_token_id).sum()),
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": time.monotonic()-started}
            report["steps"].append(item)
            save()
            print(json.dumps(item), flush=True)
        adapter = output / "adapter"
        model.save_pretrained(adapter)
        report.update(status="complete", adapter_files={str(p.relative_to(adapter)): digest(p) for p in adapter.rglob("*") if p.is_file()})
        save()
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--order", choices=("class_smoke", "seeded"), default="class_smoke")
    args = parser.parse_args()
    run(args.corpus, args.checkpoint, args.output, args.steps, args.order)
