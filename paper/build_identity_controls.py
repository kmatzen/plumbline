"""Generated clear-target controls with known shapes, not natural-scene labels."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from paper.build_reference_free_review import digest


def build(output):
    if output.exists():
        raise ValueError("preserve existing controls")
    output.mkdir(parents=True)
    scene = np.full((480, 640, 3), 245, np.uint8)
    cv2.fillPoly(scene, [np.array([[35, 330], [155, 100], [275, 330]])], (40, 130, 240))
    cv2.rectangle(scene, (290, 140), (430, 330), (210, 80, 40), -1)
    cv2.circle(scene, (535, 235), 85, (140, 55, 150), -1)
    targets = [("triangle", [155, 230]), ("rectangle", [360, 235]), ("circle", [535, 235])]
    cases = []
    for index, (shape, point) in enumerate(targets, 1):
        canvas = np.full((550, 960, 3), 24, np.uint8)
        canvas[60:540, :640] = scene
        x, y = point
        canvas[60:380, 640:] = cv2.resize(
            scene[y - 40 : y + 40, x - 40 : x + 40], (320, 320), interpolation=cv2.INTER_NEAREST
        )
        cv2.circle(canvas, (x, y + 60), 9, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (800, 220), 16, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "full_image",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "detail_crop",
            (660, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        path = output / f"control-{index}.png"
        cv2.imwrite(str(path), canvas)
        cases.append(
            {
                "name": path.name,
                "image_sha256": digest(path),
                "target_shape": shape,
                "target_center_xy": point,
            }
        )
    result = {
        "scope": "generated clear-target identity sanity controls; not real scenes, staticness or camera DNF",
        "script_sha256": digest(Path(__file__)),
        "opencv_version": cv2.__version__,
        "cases": cases,
        "policy": "same source-identity prompt; target labels exist only in this manifest, not in model inputs",
    }
    (output / "controls.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
