"""Frozen RGB-only forward/reverse tracks on explicitly selected calibration shards."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest
from paper.probe_movi_reverse_tracking import eligible, cycle_answer
from paper.stage_movi_calibration import validate_protocol


def validate_shards(root, protocol, indices):
    allowed=validate_protocol(json.loads(protocol.read_text()))
    acquisition=json.loads((root/'batch.json').read_text())
    if acquisition['protocol_sha256']!=digest(protocol) or not indices or len(indices)!=len(set(indices)) or not set(indices)<=set(allowed):
        raise ValueError('calibration-only shard selection required')
    selected=[]
    for index in indices:
        record=next((r for r in acquisition['attempts'] if r['shard_index']==index),None)
        folder=root/f'shard-{index:05d}'
        if record is None or record['status']!='complete':
            raise ValueError('wait for completed acquisition')
        path=folder/'decoded-v1/manifest.json'
        if digest(path)!=record['decoded_manifest_sha256']:
            raise ValueError('decoded manifest changed')
        decoded=json.loads(path.read_text())
        names=[s['name'] for s in decoded['scenes']]
        if decoded['status']!='complete' or decoded['shard_index']!=index or names!=record['scenes'] or len(names)!=decoded['expected_scenes']:
            raise ValueError('scene roster changed')
        if (folder/'image-evidence-v1').exists():
            raise FileExistsError('preserve prior inference: '+str(folder))
        selected.append((index,folder,names))
    return selected


def run(root, protocol, model_path, repo, checkpoint, prior_path, indices):
    import torch
    from paper.run_movi_cameras import load_rgb

    selected=validate_shards(root,protocol,indices)
    prior=json.loads(prior_path.read_text())
    if prior['status']!='complete' or digest(checkpoint)!=prior['checkpoint_sha256']:
        raise ValueError('checkpoint mismatch')
    for filename,sha in prior['source_files_sha256'].items():
        if digest(repo/filename)!=sha:
            raise ValueError('tracker source changed')
    sys.path.insert(0,str(repo))
    from cotracker.predictor import CoTrackerPredictor
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    model=CoTrackerPredictor(checkpoint=str(checkpoint),offline=True,v2=False,window_len=60).eval().cuda()
    seeds=np.array([[0,x,y] for y in range(12,256,32) for x in range(12,256,32)],np.float32)
    for index,folder,names in selected:
        output=folder/'image-evidence-v1';output.mkdir(exist_ok=False)
        result=dict(status='running',scope='fresh calibration RGB-only inference, no reference geometry or camera estimates used',
            shard_index=index,sequences=names,planned_observations=len(names)*192,
            protocol_sha256=digest(protocol),uncertainty_model_sha256=digest(model_path),
            decoded_manifest_sha256=digest(folder/'decoded-v1/manifest.json'),
            checkpoint_sha256=digest(checkpoint),source_files_sha256=prior['source_files_sha256'],
            prior_sha256=digest(prior_path),script_sha256=digest(Path(__file__)),
            scenes=[],records=[])
        def save():
            (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        save()
        try:
            for name in names:
                video,provenance=load_rgb(folder/'decoded-v1',name,names)
                batch=torch.from_numpy(video.copy()).permute(0,3,1,2)[None].float().cuda()
                with torch.inference_mode():
                    xy,seen=model(batch,queries=torch.from_numpy(seeds)[None].cuda(),backward_tracking=False)
                xy=xy[0].cpu().numpy();seen=seen[0].cpu().numpy()
                if xy.shape!=(24,64,2) or seen.shape!=(24,64) or not np.isfinite(xy).all() or seen.dtype!=bool:
                    raise ValueError('invalid forward output')
                del batch
                array_path=output/f'{name}-forward.npz'
                np.savez_compressed(array_path,xy=xy,visibility=seen)
                for frame in (12,18,23):
                    points=[dict(track=i,predicted_xy=xy[frame,i].tolist(),predicted_visible=bool(seen[frame,i])) for i in range(64)]
                    active=[r for r in points if eligible(r,256,256)]
                    returned={}
                    if active:
                        batch=torch.from_numpy(video[:frame+1][::-1].copy()).permute(0,3,1,2)[None].float().cuda()
                        queries=torch.tensor([[[0,*r['predicted_xy']] for r in active]],dtype=torch.float32,device='cuda')
                        with torch.inference_mode():
                            back,visible=model(batch,queries=queries,backward_tracking=False)
                        endpoints,visible=back[0,-1].cpu().numpy(),visible[0,-1].cpu().numpy()
                        if not np.isfinite(endpoints).all() or visible.dtype!=bool:
                            raise ValueError('invalid reverse output')
                        returned={r['track']:(endpoints[i],bool(visible[i])) for i,r in enumerate(active)}
                        del batch,queries,back
                    for point in points:
                        i=point['track'];endpoint,visible=returned.get(i,(None,False))
                        error=None if endpoint is None else float(np.linalg.norm(endpoint-seeds[i,1:]))
                        result['records'].append(dict(sequence=name,frame=frame,**point,
                            source_xy=seeds[i,1:].tolist(),reverse_attempted=i in returned,reverse_visible=visible,
                            reverse_xy=None if endpoint is None else endpoint.tolist(),cycle_error_pixels=error,
                            cycle_predicted=cycle_answer(point['predicted_visible'],visible,error)))
                    save();torch.cuda.empty_cache()
                result['scenes'].append(dict(sequence=name,provenance=provenance,forward_file=array_path.name,forward_sha256=digest(array_path)))
                save();print(index,name,len(result['records']),flush=True)
            if len(result['records'])!=result['planned_observations']:
                raise ValueError('incomplete point roster')
            result['status']='complete'
        except Exception as error:
            result.update(status='failed',error=repr(error))
            print(index,'failed',repr(error),flush=True)
        finally:
            save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','model','repo','checkpoint','prior'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--shards',nargs='+',type=int,required=True)
    args=parser.parse_args();run(args.root,args.protocol,args.model,args.repo,args.checkpoint,args.prior,args.shards)
