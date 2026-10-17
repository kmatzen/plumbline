"""Extract released normal PNGs without changing the existing decoded pilot."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper.decode_movi_geometry_pilot import digest, images


def extract(source, decoded, output):
    from tfrecord import tfrecord_iterator, example_pb2
    manifest = json.loads((decoded/'manifest.json').read_text())
    shard = source/'movi_e-validation.tfrecord-00000-of-00064'
    if manifest['status'] != 'complete' or digest(shard) != manifest['shard_sha256']:
        raise ValueError('decoded pilot or shard mismatch')
    output.mkdir(parents=True,exist_ok=False)
    result = dict(status='extracting',decoded_manifest_sha256=digest(decoded/'manifest.json'),
                  shard_sha256=digest(shard),script_sha256=digest(Path(__file__)),scenes=[])
    try:
        for index,payload in enumerate(tfrecord_iterator(str(shard))):
            example = example_pb2.Example(); example.ParseFromString(payload)
            f = example.features.feature
            name = f['metadata/video_name'].bytes_list.value[0].decode()
            if index >= len(manifest['scenes']) or name != manifest['scenes'][index]['name']:
                raise ValueError('scene roster changed')
            normals = images(f['normal'].bytes_list.value,(24,256,256,3),np.uint16,rgb=True)
            target = output/f'{index:02d}-normals.npz'
            np.savez_compressed(target,normal=normals)
            result['scenes'].append(dict(name=name,file=target.name,sha256=digest(target)))
            print(name,flush=True)
        if len(result['scenes']) != manifest['expected_scenes']:
            raise ValueError('incomplete shard')
        result['status'] = 'complete'
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','decoded','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    extract(args.source,args.decoded,args.output)
