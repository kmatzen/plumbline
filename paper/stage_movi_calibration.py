"""Acquire only the frozen calibration shards; never open reserved evaluation."""

import argparse
import json
from pathlib import Path

from paper.decode_movi_geometry_pilot import decode, digest
from paper.stage_movi_geometry_pilot import stage


def validate_protocol(plan):
    calibration=plan['calibration_shards']; evaluation=plan['reserved_evaluation_shards']
    if (plan['status'] != 'frozen_before_acquisition' or plan['split'] != 'validation'
            or calibration != list(range(2,27)) or evaluation != list(range(27,37))
            or set(calibration)&set(evaluation) or plan['expected_calibration_scenes'] != 97):
        raise ValueError('frozen calibration/evaluation split mismatch')
    return calibration


def run(protocol, model, output):
    plan=json.loads(protocol.read_text())
    shards=validate_protocol(plan)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='calibration acquisition only; reserved evaluation unopened',
                protocol_sha256=digest(protocol),model_sha256=digest(model),
                script_sha256=digest(Path(__file__)),
                helper_sha256={n:digest(Path(__file__).with_name(n)) for n in ('stage_movi_geometry_pilot.py','decode_movi_geometry_pilot.py')},
                planned_shards=shards,expected_scenes=plan['expected_calibration_scenes'],attempts=[])
    def save():
        (output/'batch.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    names=set();records=set()
    for index in shards:
        folder=output/f'shard-{index:05d}'
        attempt=dict(shard_index=index,directory=folder.name,status='acquiring')
        report['attempts'].append(attempt);save()
        try:
            stage(folder,index)
            attempt['status']='decoding';save()
            decoded=decode(folder,folder/'decoded-v1')
            new_names={r['name'] for r in decoded['scenes']}
            new_records={r['record_sha256'] for r in decoded['scenes']}
            if names&new_names or records&new_records:
                raise ValueError('duplicate scenes across calibration shards')
            names.update(new_names);records.update(new_records)
            attempt.update(status='complete',scenes=[r['name'] for r in decoded['scenes']],
                           staged_manifest_sha256=digest(folder/'manifest.json'),
                           decoded_manifest_sha256=digest(folder/'decoded-v1/manifest.json'))
        except Exception as error:
            attempt.update(status='failed',error=repr(error))
        save();print(index,attempt['status'],len(names),flush=True)
    report.update(status='complete' if all(a['status']=='complete' for a in report['attempts']) and len(names)==report['expected_scenes'] else 'finished_with_incomplete_inputs',
                  decoded_unique_scenes=len(names))
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','model','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args();run(args.protocol,args.model,args.output)
