"""Run each predeclared camera attempt in a fresh process and retain failures."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(root, plan_path, dage_checkpoint, da3_checkpoint, output):
    if output.exists():
        raise ValueError("preserve existing batch")
    plan = json.loads(plan_path.read_text())
    source = json.loads((root/"manifest.json").read_text())
    names = sorted(n for n, e in source["sequences"].items() if e["split"] == "held_out")
    if len(names) != 20 or plan["sequences"] != names or plan["methods"] != ["dage", "da3"]:
        raise ValueError("fixed twenty-video/two-method roster required")
    output.mkdir(parents=True)
    report = dict(status="running", planned_attempts=40, attempts=[])

    def save():
        temp = output/"batch.tmp"
        temp.write_text(json.dumps(report, indent=2)+"\n")
        temp.replace(output/"batch.json")

    save()
    for name in names:
        for method, checkpoint in (("dage", dage_checkpoint), ("da3", da3_checkpoint)):
            identity = f"{name}-{method}"
            row = dict(sequence=name, method=method, status="running", output=identity)
            report["attempts"].append(row)
            save()
            with (output/f"{identity}.log").open("x") as log:
                result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_tapvid_cameras.py")),
                    str(root), name, method, str(checkpoint), str(output/identity)], stdout=log, stderr=subprocess.STDOUT)
            row.update(status="process_complete" if result.returncode == 0 else "process_failed", returncode=result.returncode)
            save()
            print(identity, row["status"], flush=True)
    report["status"] = "finished_all_attempts"
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "plan", "dage_checkpoint", "da3_checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.root, args.plan, args.dage_checkpoint, args.da3_checkpoint, args.output)
