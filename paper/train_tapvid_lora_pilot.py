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


def memorization_schedule(rows, repeats):
    if repeats != 20:
        raise ValueError("memorization diagnostic has a fixed 20-pass budget")
    selected = []
    for label in ("close", "far", "occluded"):
        candidates = sorted((r for r in rows if r["target"] == label), key=lambda r: r["id"])
        if len(candidates) < 3:
            raise ValueError("three training examples per class required")
        selected.extend(candidates[:3])
    schedule = []
    for epoch in range(repeats):
        block = selected.copy()
        random.Random(epoch).shuffle(block)
        schedule.extend(block)
    return selected, schedule


def repeat_schedule(rows, epochs):
    if epochs not in {1, 20}:
        raise ValueError("only the fixed one-pass or twenty-pass study is supported")
    schedule = rows.copy()
    for epoch in range(1, epochs):
        block = rows.copy()
        random.Random(epoch).shuffle(block)
        schedule.extend(block)
    return schedule


def run(corpus, checkpoint, output, steps, order="class_smoke", probe_index=None, compute_dtype="float16", memorization=False, epochs=1):
    if compute_dtype not in {"float16", "float32"}:
        raise ValueError("unsupported quantized compute dtype")
    if output.exists():
        raise ValueError("preserve prior training runs")
    rows = select_training(corpus, steps, order)
    training_ids = [r["id"] for r in rows]
    if epochs != 1:
        if memorization or probe_index is not None or order != "seeded":
            raise ValueError("full repeated study requires seeded order and no diagnostic subset")
        full_count = sum(r["partition"] == "train" and r["status"] == "labeled"
                         for r in json.loads((corpus / "manifest.json").read_text())["records"])
        if len(rows) != full_count:
            raise ValueError("repeated study requires the complete training pool")
        rows = repeat_schedule(rows, epochs)
    diagnostic_rows = None
    if memorization:
        if probe_index is not None:
            raise ValueError("cannot combine memorization and arithmetic probe")
        full_count = sum(r["partition"] == "train" and r["status"] == "labeled"
                         for r in json.loads((corpus / "manifest.json").read_text())["records"])
        if len(rows) != full_count:
            raise ValueError("memorization selection requires the complete training pool")
        diagnostic_rows, rows = memorization_schedule(rows, 20)
    if probe_index is not None:
        if not 0 <= probe_index < len(rows):
            raise ValueError("probe index outside selected training order")
        rows = [rows[probe_index]]
    output.mkdir(parents=True)
    report = {"status": "starting", "scope": "training feasibility only, not generalization or camera evaluation",
              "training_ids": [r["id"] for r in rows], "corpus_manifest_sha256": digest(corpus / "manifest.json"),
              "train_export_sha256": digest(corpus / "train.jsonl"), "script_sha256": digest(Path(__file__)),
              "checkpoint": str(checkpoint), "steps_requested": len(rows), "selection_pool_steps": steps,
              "diagnostic_probe_index": probe_index, "steps": [],
              "order": order,
              "policy": "training only; seed 0; one pass maximum; no calibration/validation files read; class_smoke prioritizes one per class; seeded shuffles every training example",
              "configuration": {"rank": 8, "alpha": 16, "dropout": 0., "learning_rate": 1e-4,
                                "quantized_compute_dtype": compute_dtype,
                                "target_modules": ["q_proj", "v_proj"], "visual_min_tokens": 576,
                                "visual_max_tokens": 768, "batch_size": 1, "gradient_clip": 1.}}
    if memorization:
        report.update(scope="nine-example training-only memorization diagnostic; not generalization or camera evaluation",
                      memorization_ids=[r["id"] for r in diagnostic_rows],
                      policy="first three IDs per training class; 20 passes; each pass shuffled with its zero-based index as seed; no held-out examples or weight selection",
                      memorization_predictions=[])
    elif epochs != 1:
        report.update(training_ids=training_ids, update_ids=[r["id"] for r in rows],
                      epochs=epochs,
                      policy="twenty full passes; first pass identical to seed-0 one-pass run; later passes shuffle that order with pass index as seed; no resampling, held-out reads, early stopping, or checkpoint selection")

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
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, compute_dtype), bnb_4bit_use_double_quant=True))
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
        report.update(adapter_files={str(p.relative_to(adapter)): digest(p) for p in adapter.rglob("*") if p.is_file()})
        if diagnostic_rows is not None:
            report["status"] = "checking_training_fit"
            save()
            model.gradient_checkpointing_disable()
            model.eval()
            with torch.inference_mode():
                for row in diagnostic_rows:
                    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}
                    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
                    with Image.open(corpus / row["image"]) as im:
                        inputs = processor(text=[prompt], images=[im.convert("RGB")], return_tensors="pt")
                    pixel_sha = hashlib.sha256(inputs["pixel_values"].numpy().tobytes()).hexdigest()
                    inputs = inputs.to("cuda")
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True,
                                               return_dict_in_generate=True, output_scores=True)
                    if any(torch.isnan(s).any() or torch.isposinf(s).any() for s in generated.scores):
                        raise FloatingPointError("nonfinite diagnostic generation scores")
                    raw = processor.batch_decode(generated.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                    parsed = raw.strip().lower()
                    if parsed not in {"close", "far", "occluded", "uncertain"}:
                        parsed = "invalid"
                    item = dict(id=row["id"], target_for_scoring_only=row["target"], raw_response=raw,
                                parsed=parsed, processed_pixel_sha256=pixel_sha)
                    report["memorization_predictions"].append(item)
                    save()
                    print(json.dumps(item), flush=True)
        report["status"] = "complete"
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
    parser.add_argument("--memorization", action="store_true", help="fixed nine-example, twenty-pass training-only diagnostic")
    parser.add_argument("--epochs", type=int, choices=(1, 20), default=1)
    parser.add_argument("--order", choices=("class_smoke", "seeded"), default="class_smoke")
    parser.add_argument("--probe-index", type=int, help="diagnose one indexed training example from the unchanged base")
    parser.add_argument("--compute-dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    run(args.corpus, args.checkpoint, args.output, args.steps, args.order, args.probe_index, args.compute_dtype, args.memorization, args.epochs)
