"""Stage the first MOVi-E validation shard for geometry development only."""

import argparse
import base64
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

PREFIX = "tfds/movi_e/256x256/1.0.0/"
FILES = ("dataset_info.json", "features.json", "instances-category.labels.txt",
         "movi_e-validation.tfrecord-00000-of-00064")


def validate_download(path, metadata):
    with path.open("rb") as stream:
        md5 = base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode()
    if path.stat().st_size != int(metadata["size"]) or md5 != metadata["md5Hash"]:
        raise ValueError("download size or publisher checksum mismatch")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage(output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="staging", dataset="MOVi-E/256x256/1.0.0",
                    scope="first validation shard for geometry and parser development; not held-out evaluation",
                    selection="shard 00000 selected before decoding any examples; keep every contained scene",
                    policy="no test shards; no candidate camera inference or DNF labels; keep RGB separate from reference annotations in downstream use",
                    files=[])
    path = output/"manifest.json"
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    try:
        for name in FILES:
            key = PREFIX+name
            encoded = urllib.parse.quote(key, safe="")
            metadata_url = f"https://storage.googleapis.com/storage/v1/b/kubric-public/o/{encoded}"
            with urllib.request.urlopen(metadata_url, timeout=60) as response:
                metadata = json.load(response)
            if metadata["name"] != key or int(metadata["size"]) > 150_000_000:
                raise ValueError("unexpected object or excessive download size")
            url = f"https://storage.googleapis.com/kubric-public/{key}?generation={metadata['generation']}"
            record = dict(file=name, url=url, metadata_url=metadata_url,
                          generation=metadata["generation"], size=int(metadata["size"]), md5Hash=metadata["md5Hash"])
            manifest["files"].append(record)
            path.write_text(json.dumps(manifest, indent=2)+"\n")
            with urllib.request.urlopen(url, timeout=60) as source, (output/name).open("xb") as target:
                shutil.copyfileobj(source, target)
            record["sha256"] = validate_download(output/name, metadata)
            print(name, record["size"], record["sha256"], flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    stage(parser.parse_args().output)
