"""Counterfactual image-input audit on three training examples; not accuracy."""

import argparse
import importlib.util
import json
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest


def run(corpus, training, output):
    import torch
    from PIL import Image
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as implementation

    train = json.loads((training/'manifest.json').read_text())
    manifest = json.loads((corpus/'manifest.json').read_text())
    assert train['status'] == 'complete'
    assert digest(corpus/'manifest.json') == train['corpus_manifest_sha256']
    assert digest(corpus/'train.jsonl') == manifest['exports_sha256']['train']
    rows = [json.loads(line) for line in (corpus/'train.jsonl').read_text().splitlines()][:3]
    checkpoint = Path(train['checkpoint'])
    for name, sha in train['checkpoint_files'].items():
        assert digest(checkpoint/name) == sha
    for name, sha in train['adapter_files'].items():
        assert digest(training/'adapter'/name) == sha
    helper = Path(__file__).with_name('fp32_attention_scores.py')
    assert digest(helper) == train['attention_helper_sha256']
    spec = importlib.util.spec_from_file_location('audit_fp32', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = implementation.eager_attention_forward
    implementation.eager_attention_forward = lambda layer, *a, **kw: (
        module.attention(layer, *a, **kw) if type(layer).__name__ == 'Qwen2_5_VLAttention'
        else original(layer, *a, **kw))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='loading', scope='image-input dependence audit on first three training rows; not accuracy, validation, or model selection',
        protocol='base/adapted; unchanged correspondence prompt; original pixels versus zero normalized pixel payload; identical text, grid and image-token layout; greedy eight tokens; first-token probability total variation',
        limitation='zero payload is an artificial intervention, not a natural image; sensitivity is necessary but not sufficient for useful visual reasoning; only three previously trained examples',
        script_sha256=digest(Path(__file__)), training_manifest_sha256=digest(training/'manifest.json'),
        corpus_manifest_sha256=digest(corpus/'manifest.json'), attention_helper_sha256=digest(helper), records=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    try:
        torch.manual_seed(0)
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True,
            min_pixels=576*28*28, max_pixels=768*28*28)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='eager', device_map={'':0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=getattr(torch, train['configuration']['quantized_compute_dtype']), bnb_4bit_use_double_quant=True))
        model = PeftModel.from_pretrained(prepare_model_for_kbit_training(base, use_gradient_checkpointing=False),
            training/'adapter', is_trainable=False).eval()
        import contextlib
        import hashlib
        result['status'] = 'evaluating'; save()
        for arm in ('base', 'adapted'):
            with (model.disable_adapter() if arm == 'base' else contextlib.nullcontext()), torch.inference_mode():
                for row in rows:
                    entry = next(r for r in manifest['records'] if r['id'] == row['id'])
                    assert entry['partition'] == 'train'
                    assert digest(corpus/row['image']) == entry['image_sha256']
                    messages = [{'role':'user', 'content':[{'type':'image'}, {'type':'text','text':row['prompt']}]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    with Image.open(corpus/row['image']) as im:
                        inputs = processor(text=[text], images=[im.convert('RGB')], return_tensors='pt').to('cuda')
                    tokens = int((inputs['input_ids'] == model.config.image_token_id).sum())
                    assert tokens == int(inputs['image_grid_thw'].prod())//processor.image_processor.merge_size**2 > 0
                    pixel = inputs['pixel_values'].clone()
                    record = dict(arm=arm, id=row['id'], image_sha256=entry['image_sha256'], image_tokens=tokens, conditions={})
                    probabilities = []
                    for condition in ('original', 'zero_pixels'):
                        inputs['pixel_values'] = pixel if condition == 'original' else torch.zeros_like(pixel)
                        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True,
                            return_dict_in_generate=True, output_scores=True)
                        assert all(torch.isfinite(s).all() for s in generated.scores)
                        probabilities.append(generated.scores[0].float().softmax(-1).cpu())
                        record['conditions'][condition] = dict(
                            pixel_sha256=hashlib.sha256(inputs['pixel_values'].cpu().numpy().tobytes()).hexdigest(),
                            raw_response=processor.batch_decode(generated.sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0])
                    record['first_token_total_variation'] = float((probabilities[0]-probabilities[1]).abs().sum()/2)
                    result['records'].append(record); save()
                    print(arm, row['id'], record['first_token_total_variation'], record['conditions'], flush=True)
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'training', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.corpus, args.training, args.output)
