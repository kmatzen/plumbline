"""Run the prospectively selected replication roster with frozen camera settings."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest


def roster(root, protocol):
    plan = json.loads(protocol.read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    names = [s['name'] for s in manifest['scenes']]
    if (plan['status'] != 'frozen_before_shard_acquisition' or plan['split'] != 'validation'
            or plan['shard_index'] != 1 or manifest['shard_index'] != 1
            or manifest['status'] != 'complete' or len(names) != manifest['expected_scenes']
            or len(names) != len(set(names)) or not names or plan['camera_methods'] != ['dage', 'da3']):
        raise ValueError('replication roster or protocol mismatch')
    return names


def run(root, protocol, prior_runs, output):
    names = roster(root, protocol)
    cameras = {}
    for method in ('dage', 'da3'):
        prior_path = prior_runs/f'3835-{method}'/'manifest.json'
        prior = json.loads(prior_path.read_text())
        if prior['status'] != 'complete' or prior['method'] != method:
            raise ValueError('completed prior run required')
        checkpoint = Path(prior['checkpoint'])
        for filename, sha in prior['checkpoint_files'].items():
            if digest(checkpoint/filename) != sha:
                raise ValueError('checkpoint changed')
        cameras[method] = dict(prior_manifest_sha256=digest(prior_path), checkpoint=str(checkpoint),
                               checkpoint_files=prior['checkpoint_files'], settings=prior['settings'])
    output.mkdir(parents=True, exist_ok=False)
    batch = dict(status='running', scope='prospective validation-shard replication; all attempts retained',
                 protocol_sha256=digest(protocol), input_manifest_sha256=digest(root/'manifest.json'),
                 script_sha256=digest(Path(__file__)),
                 runner_sha256=digest(Path(__file__).with_name('run_movi_cameras.py')),
                 sequences=names, cameras=cameras, attempts=[])
    def save():
        (output/'batch.json').write_text(json.dumps(batch, indent=2)+'\n')
    save()
    for name in names:
        for method in ('dage', 'da3'):
            directory = output/f'{name}-{method}'
            row = dict(sequence=name, method=method, output=directory.name, status='running')
            batch['attempts'].append(row)
            save()
            with (output/f'{name}-{method}.log').open('x') as log:
                code = subprocess.call([sys.executable, '-m', 'paper.run_movi_cameras', str(root), name, method,
                    cameras[method]['checkpoint'], str(directory), '--sequences', *names],
                    stdout=log, stderr=subprocess.STDOUT)
            row.update(returncode=code, status='process_complete' if code == 0 else 'process_failed')
            manifest_path = directory/'manifest.json'
            if manifest_path.exists():
                result = json.loads(manifest_path.read_text())
                row['manifest_sha256'] = digest(manifest_path)
                row['camera_status'] = result['status']
                if result['status'] == 'complete':
                    row['matches_prior_settings'] = result['settings'] == cameras[method]['settings']
                    row['matches_prior_checkpoint'] = result['checkpoint_files'] == cameras[method]['checkpoint_files']
                    row['matches_prior_model_sources'] = result['source_files'] == json.loads(
                        (prior_runs/f'3835-{method}'/'manifest.json').read_text())['source_files']
            save()
            print(name, method, row['status'], row.get('camera_status'), flush=True)
    batch['status'] = 'finished_all_attempts'
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'prior_runs', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.root, args.protocol, args.prior_runs, args.output)
