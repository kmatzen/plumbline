"""RGB-only camera inference on the four fixed development pilot scenes."""

import argparse
import hashlib
import inspect
import json
import time
import traceback
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.query_chunk_attention import query_chunk_attention
from paper.run_davis_da3 import normalized_input_intrinsics


def load_rgb(root,name):
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['status'] != 'complete' or manifest['expected_scenes'] != 4
            or [s['name'] for s in manifest['scenes']] != ['3835','5397','4385','4970']):
        raise ValueError('complete pilot roster required')
    entries = [s for s in manifest['scenes'] if s['name'] == name]
    if len(entries) != 1:
        raise ValueError('scene absent or duplicated')
    entry = entries[0]
    path = (root/entry['rgb_file']).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != entry['rgb_sha256']:
        raise ValueError('RGB path or hash mismatch')
    with np.load(path,allow_pickle=False) as archive:
        if archive.files != ['video']:
            raise ValueError('RGB-only archive required')
        video = archive['video']
    sha = hashlib.sha256(video.tobytes()).hexdigest()
    if video.shape != (24,256,256,3) or video.dtype != np.uint8 or sha != entry['decoded_rgb_sha256']:
        raise ValueError('decoded RGB mismatch')
    return video,dict(input_manifest_sha256=digest(manifest_path),rgb_file_sha256=digest(path),
                     decoded_rgb_sha256=sha,frame_sha256=[hashlib.sha256(f.tobytes()).hexdigest() for f in video])


def run(root,name,method,checkpoint,output):
    if method not in {'dage','da3'}:
        raise ValueError('unknown camera method')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='preparing',sequence=name,method=method,seed=0,
        scope='four-scene MOVi-E development camera outputs, not held-out validation',
        input_policy='RGB-only NPZ; no reference poses, intrinsics, depths, masks, normals, or object motion loaded',
        script_sha256=digest(Path(__file__)))

    def save():
        (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

    save()
    try:
        video,provenance = load_rgb(root,name)
        import torch
        from plumbline.models.dage import DAGEAdapter
        from plumbline.models.depth_anything_3 import DepthAnything3Adapter
        from plumbline.conventions import assert_valid_extrinsics
        required = ('model.pt',) if method == 'dage' else ('config.json','model.safetensors')
        package = Path(inspect.getfile(DAGEAdapter)).parents[1]
        report.update(status='running',**provenance,checkpoint=str(checkpoint),
            checkpoint_files={n:digest(checkpoint/n) for n in required},
            source_files={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*.py'))},
            helpers_sha256={n:digest(Path(__file__).with_name(n)) for n in ('query_chunk_attention.py','run_davis_da3.py','decode_movi_geometry_pilot.py')},
            torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
        torch.manual_seed(0); np.random.seed(0); torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        if method == 'dage':
            settings = dict(lr_max_size=252,hr_max_size=252,compute_dtype='float16')
            report['settings'] = settings; save()
            model = DAGEAdapter(checkpoint=str(checkpoint/'model.pt'),**settings)
            model._load(); native_infer = model._model.infer; captured = {}

            def retain(*args,**kwargs):
                result = native_infer(*args,**kwargs)
                captured['k'] = np.asarray(result['intrinsics'].detach().float().cpu(),dtype=np.float32)
                return result

            model._model.infer = retain
            prediction = model.predict(video,intrinsics=None)
            calibration = captured['k']
        else:
            report.update(settings=dict(process_res=252,process_res_method='upper_bound_resize',query_chunk=256),
                          attention_caveat='end-to-end equivalence to unchunked inference not established')
            save()
            model = DepthAnything3Adapter(checkpoint=str(checkpoint),process_res=252)
            with query_chunk_attention(256):
                prediction = model.predict(video)
            calibration = normalized_input_intrinsics(prediction.intrinsics,prediction.depth.shape[1:],video.shape[1:3])
        torch.cuda.synchronize()
        if calibration.shape != (24,3,3) or not np.isfinite(calibration).all() or prediction.extrinsics.shape != (24,4,4):
            raise ValueError('invalid camera output')
        assert_valid_extrinsics(prediction.extrinsics)
        np.savez_compressed(output/'prediction.npz',extrinsics=prediction.extrinsics,intrinsics_normalized=calibration)
        report.update(status='complete',prediction_sha256=digest(output/'prediction.npz'),
            prediction_metadata=prediction.metadata,calibration='predicted normalized intrinsics; raster=normalized*size-0.5',
            elapsed_seconds_including_load=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path); parser.add_argument('sequence')
    parser.add_argument('method',choices=('dage','da3'))
    parser.add_argument('checkpoint',type=Path); parser.add_argument('output',type=Path)
    args = parser.parse_args()
    run(args.root,args.sequence,args.method,args.checkpoint,args.output)
