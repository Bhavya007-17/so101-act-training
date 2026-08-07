#!/usr/bin/env python
"""Step-1 gate: prove offscreen rendering actually works on this WSL2 box.

MUJOCO_GL must be set *before* mujoco is imported, so this script renders under
exactly one backend per process and `render_probe.sh` runs it once per candidate.

The check is deliberately paranoid. A dead GL backend on WSL2 does not always
raise -- it can hand back a buffer that is uniformly black, uniformly grey, or
uninitialised garbage. So a render only counts as passing if the frame has real
structure in it: non-trivial standard deviation, and many distinct colours.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

BACKEND = os.environ.get("MUJOCO_GL", "<unset>")

import mujoco  # noqa: E402  (import order is load-bearing: see module docstring)
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
SCENE = REPO / "sim" / "assets" / "so101" / "scene_pickplace.xml"

# Recording geometry. 480x640 matches the SO-101 dataset this repo already
# trained ACT on, so the measured batch-size ladder in DECISIONS.md stays valid.
HEIGHT, WIDTH = 480, 640
CAMERAS = ("external", "wrist_cam")


def frame_stats(img: np.ndarray) -> dict:
    """Structure metrics for one RGB frame. Cheap, and enough to catch a dead buffer."""
    flat = img.reshape(-1, img.shape[-1])
    return {
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "min": int(img.min()),
        "max": int(img.max()),
        "mean": round(float(img.mean()), 3),
        "std": round(float(img.std()), 3),
        "unique_colors": int(len(np.unique(flat, axis=0))),
        "frac_pure_black": round(float((flat.sum(axis=1) == 0).mean()), 4),
    }


def verdict(stats: dict) -> tuple[bool, str]:
    """A frame passes only if it looks like a rendered scene, not a dead buffer."""
    if stats["max"] == 0:
        return False, "frame is entirely black (max pixel value 0)"
    if stats["std"] < 1.0:
        return False, f"frame is near-uniform (std={stats['std']}) -- likely a dead buffer"
    if stats["unique_colors"] < 50:
        return False, f"only {stats['unique_colors']} distinct colours -- likely garbage"
    if stats["frac_pure_black"] > 0.98:
        return False, f"{stats['frac_pure_black']:.1%} of pixels are pure black"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "outputs" / "render_probe"))
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"MUJOCO_GL       : {BACKEND}")
    print(f"mujoco version  : {mujoco.__version__}")
    print(f"scene           : {SCENE}")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    # Land on the documented home keyframe rather than qpos0, so the probe image
    # shows the pose the episode will actually start from.
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    results: dict[str, dict] = {}
    all_ok = True

    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for cam in CAMERAS:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cam_id < 0:
                print(f"[FAIL] camera {cam!r} does not exist in the model")
                all_ok = False
                continue

            renderer.update_scene(data, camera=cam)
            img = renderer.render()

            stats = frame_stats(img)
            ok, why = verdict(stats)
            all_ok &= ok

            png = outdir / f"probe_{cam}.png"
            _write_png(img, png)

            results[cam] = {**stats, "ok": ok, "why": why, "png": str(png)}
            tag = "PASS" if ok else "FAIL"
            print(
                f"[{tag}] {cam:<10} {stats['shape']} "
                f"mean={stats['mean']:<7} std={stats['std']:<7} "
                f"colours={stats['unique_colors']:<6} -> {why}"
            )
            print(f"       wrote {png}")

    (outdir / "probe_stats.json").write_text(
        json.dumps({"backend": BACKEND, "cameras": results}, indent=2)
    )

    print(f"\nRESULT: {'RENDERING VERIFIED' if all_ok else 'RENDERING BROKEN'} (backend={BACKEND})")
    return 0 if all_ok else 1


def _write_png(img: np.ndarray, path: pathlib.Path) -> None:
    import imageio.v3 as iio

    iio.imwrite(path, img)


if __name__ == "__main__":
    sys.exit(main())
