"""Frozen validation comparison or training-fit diagnostic; never calibration."""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import time
import traceback
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def parse_label(text):
    label = text.strip().lower()
    return label if label in {"close", "far", "occluded", "uncertain"} else "invalid"


def select_evaluation(corpus, train, manifest, partition):
    if partition not in {"train", "validation"}:
        raise ValueError("only training-fit or validation evaluation is allowed")
    source = corpus / f"{partition}.jsonl"
    if digest(source) != manifest["exports_sha256"][partition]:
        raise ValueError("evaluation export changed")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    allowed = {r["id"]: r for r in manifest["records"] if r["partition"] == partition and r["status"] == "labeled"}
    if {r["id"] for r in rows} != set(allowed) or len(rows) != len(allowed):
        raise ValueError("evaluation membership changed")
    if partition == "validation" and set(train["training_ids"]) & set(allowed):
        raise ValueError("training/validation ID overlap")
    if partition == "train" and (set(train["training_ids"]) != set(allowed)
                                  or len(train["training_ids"]) != len(allowed)):
        raise ValueError("training-fit requires the complete training partition")
    for row in rows:
        item = allowed[row["id"]]
        if row["target"] != item["label"] or row["image"] != item["image"] or row["prompt"] != manifest["prompt"]:
            raise ValueError("evaluation content changed")
        if digest(corpus / row["image"]) != item["image_sha256"]:
            raise ValueError("evaluation image changed")
    return source, rows, allowed


def run(corpus, training_run, output, partition="validation"):
    if output.exists():
        raise ValueError("preserve previous comparison")
    train = json.loads((training_run / "manifest.json").read_text())
    manifest = json.loads((corpus / "manifest.json").read_text())
    if train["status"] != "complete" or train["corpus_manifest_sha256"] != digest(corpus / "manifest.json"):
        raise ValueError("completed training run with identical corpus required")
    source, rows, allowed = select_evaluation(corpus, train, manifest, partition)
    arms = ("base", "adapted") if partition == "validation" else ("adapted",)
    checkpoint = Path(train["checkpoint"])
    for name, sha in train["checkpoint_files"].items():
        if digest(checkpoint / name) != sha:
            raise ValueError("base checkpoint changed")
    for name, sha in train["adapter_files"].items():
        if digest(training_run / "adapter" / name) != sha:
            raise ValueError("adapter changed")
    output.mkdir(parents=True)
    report = {"status": "starting", "scope": ("development validation comparison, not camera DNF or an external test" if partition == "validation" else "training-fit diagnostic only; not generalization or model selection"),
              "partition": partition, "evaluation_export_sha256": digest(source),
              "training_manifest_sha256": digest(training_run / "manifest.json"),
              "script_sha256": digest(Path(__file__)),
              "protocol": "validation: base with adapters disabled versus adapted; training-fit: adapted only; identical prepared quantized model and original panels/prompts; greedy max 8 new tokens; exact four-label parser; invalid and uncertain retained; no calibration examples read; no weight updates",
              "results": {arm: [] for arm in arms}}
    if partition == "validation":
        report["validation_export_sha256"] = digest(source)

    def save():
        (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    save()
    try:
        import torch
        from PIL import Image
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as implementation

        helper = Path(__file__).with_name("fp32_attention_scores.py")
        if digest(helper) != train["attention_helper_sha256"]:
            raise ValueError("training/evaluation attention helper changed")
        spec = importlib.util.spec_from_file_location("eval_fp32_scores", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = implementation.eager_attention_forward

        def attention(layer, *args, **kwargs):
            return module.attention(layer, *args, **kwargs) if type(layer).__name__ == "Qwen2_5_VLAttention" else original(layer, *args, **kwargs)

        implementation.eager_attention_forward = attention
        torch.manual_seed(0)
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, min_pixels=576*28*28, max_pixels=768*28*28)
        compute_dtype = train["configuration"].get("quantized_compute_dtype", "float16")
        if compute_dtype not in {"float16", "float32"}:
            raise ValueError("unknown training arithmetic")
        report["quantized_compute_dtype"] = compute_dtype
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.float16, attn_implementation="eager",
            device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, compute_dtype), bnb_4bit_use_double_quant=True))
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(base, training_run / "adapter", is_trainable=False)
        model.eval()
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("evaluation model has trainable weights")
        report["status"] = "evaluating"
        save()
        for arm in arms:
            manager = model.disable_adapter() if arm == "base" else contextlib.nullcontext()
            with manager, torch.inference_mode():
                for row in rows:
                    started = time.monotonic()
                    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    with Image.open(corpus / row["image"]) as im:
                        inputs = processor(text=[text], images=[im.convert("RGB")], return_tensors="pt")
                    pixel = inputs["pixel_values"]
                    if not torch.isfinite(pixel).all():
                        raise ValueError("nonfinite image payload")
                    pixel_sha = hashlib.sha256(pixel.numpy().tobytes()).hexdigest()
                    image_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum())
                    expected = int(inputs["image_grid_thw"].prod()) // processor.image_processor.merge_size**2
                    if image_tokens != expected or image_tokens == 0:
                        raise ValueError("image token mismatch")
                    inputs = inputs.to("cuda")
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                               use_cache=True, return_dict_in_generate=True, output_scores=True)
                    if any(torch.isnan(s).any() or torch.isposinf(s).any() for s in generated.scores):
                        raise FloatingPointError("nonfinite generation scores")
                    raw = processor.batch_decode(generated.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                    item = {"id": row["id"], "sequence": allowed[row["id"]]["sequence"],
                            "target_for_scoring_only": row["target"], "raw_response": raw, "parsed": parse_label(raw),
                            "processed_pixel_sha256": pixel_sha, "image_tokens": image_tokens,
                            "elapsed_seconds": time.monotonic()-started}
                    report["results"][arm].append(item)
                    save()
                    print(arm, len(report["results"][arm]), row["id"], item["parsed"], flush=True)
        if partition == "validation":
            for a, b in zip(report["results"]["base"], report["results"]["adapted"], strict=True):
                if a["id"] != b["id"] or a["processed_pixel_sha256"] != b["processed_pixel_sha256"]:
                    raise ValueError("paired inputs changed")
        report["status"] = "complete"
        save()
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "training_run", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--partition", choices=("train", "validation"), default="validation")
    args = parser.parse_args()
    run(args.corpus, args.training_run, args.output, args.partition)
