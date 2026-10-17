"""Run all eight fixed RGB-only development attempts, retaining failures."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest


def run(root,plan_path,dage,da3,output):
    plan=json.loads(plan_path.read_text())
    if (plan['sequences'] != ['3835','5397','4385','4970'] or plan['methods'] != ['dage','da3']
            or plan['input_manifest_sha256'] != digest(root/'manifest.json')):
        raise ValueError('fixed pilot protocol mismatch')
    output.mkdir(parents=True,exist_ok=False)
    batch=dict(status='running',protocol_sha256=digest(plan_path),script_sha256=digest(Path(__file__)),
               runner_sha256=digest(Path(__file__).with_name('run_movi_cameras.py')),attempts=[])
    def save():
        (output/'batch.json').write_text(json.dumps(batch,indent=2)+'\n')
    save()
    for sequence in plan['sequences']:
        for method,checkpoint in (('dage',dage),('da3',da3)):
            name=f'{sequence}-{method}'
            row=dict(sequence=sequence,method=method,status='running',output=name)
            batch['attempts'].append(row); save()
            with (output/f'{name}.log').open('x') as log:
                code=subprocess.call([sys.executable,'-m','paper.run_movi_cameras',str(root),sequence,method,
                                      str(checkpoint),str(output/name)],stdout=log,stderr=subprocess.STDOUT)
            row.update(returncode=code,status='process_complete' if code==0 else 'process_failed')
            save(); print(name,row['status'],flush=True)
    batch['status']='finished_all_attempts'; save()
    return batch


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','plan','dage','da3','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    run(args.root,args.plan,args.dage,args.da3,args.output)
