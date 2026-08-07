#!/usr/bin/env python
"""Eyeball harness: make a recorded episode actually lookable-at.

Reads back the episode *from the written LeRobot dataset*, not from memory, so
what you inspect is what was persisted -- including the video round-trip. Emits:

  frames/<camera>/NNNN.png   every recorded frame, both cameras
  contact_sheet_<camera>.png a grid of evenly-spaced frames
  episode_<camera>.gif       the whole episode as an animation
  trajectories.png           joint positions + commanded actions, phases shaded

    MUJOCO_GL=egl uv run python scripts/inspect_episode.py --seed 20260807
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sim import config as C  # noqa: E402

PHASE_COLORS = {
    "approach": "#4C78A8",
    "grasp": "#F58518",
    "transport": "#54A24B",
    "release": "#E45756",
}


def load(root: pathlib.Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)
    meta = json.loads((root / "episode_metadata.json").read_text())
    return dataset, meta


def to_uint8(t) -> np.ndarray:
    """LeRobot returns images as float CHW in [0,1]; PNGs want uint8 HWC."""
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255).round().astype(np.uint8)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--root", default=None)
    ap.add_argument("--repo-id", default="local/so101_sim_pickplace")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--contact-cols", type=int, default=8)
    ap.add_argument("--contact-rows", type=int, default=4)
    args = ap.parse_args()

    root = pathlib.Path(args.root or C.REPO / "outputs" / "episodes" / f"seed_{args.seed}")
    outdir = pathlib.Path(args.outdir or C.REPO / "outputs" / "inspect" / f"seed_{args.seed}")
    outdir.mkdir(parents=True, exist_ok=True)

    import imageio.v3 as iio

    dataset, meta = load(root, args.repo_id)
    n = len(dataset)
    cams = list(C.CAMERAS)
    print(f"episode  : seed {meta['seed']}  success={meta['success']}  frames={n}")
    print(f"phases   : {meta['frames_per_phase']}")

    stacks = {c: [] for c in cams}
    states, actions = [], []

    for i in range(n):
        item = dataset[i]
        states.append(np.asarray(item["observation.state"]))
        actions.append(np.asarray(item["action"]))
        for cam in cams:
            img = to_uint8(item[f"observation.images.{cam}"])
            stacks[cam].append(img)
            d = outdir / "frames" / cam
            d.mkdir(parents=True, exist_ok=True)
            iio.imwrite(d / f"{i:04d}.png", img)

    states = np.array(states)
    actions = np.array(actions)
    print(f"dumped   : {n * len(cams)} PNGs -> {outdir / 'frames'}")

    # --- contact sheets + GIFs ---------------------------------------------
    for cam in cams:
        frames = stacks[cam]
        k = min(args.contact_cols * args.contact_rows, len(frames))
        picks = np.linspace(0, len(frames) - 1, k).round().astype(int)
        rows = []
        for r in range(args.contact_rows):
            row = [frames[picks[r * args.contact_cols + c]]
                   for c in range(args.contact_cols)
                   if r * args.contact_cols + c < k]
            if row:
                while len(row) < args.contact_cols:
                    row.append(np.zeros_like(row[0]))
                rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        iio.imwrite(outdir / f"contact_sheet_{cam}.png", sheet)
        iio.imwrite(outdir / f"episode_{cam}.gif", frames, duration=1000 / C.RECORD_HZ, loop=0)
        print(f"contact  : contact_sheet_{cam}.png  ({sheet.shape[1]}x{sheet.shape[0]})")
        print(f"gif      : episode_{cam}.gif")

    # --- trajectory plot ----------------------------------------------------
    t = np.arange(n) / C.RECORD_HZ
    fig, axes = plt.subplots(C.N_JOINTS, 1, figsize=(11, 13), sharex=True)

    start = 0
    spans = []
    for phase in ("approach", "grasp", "transport", "release"):
        count = meta["frames_per_phase"].get(phase, 0)
        if count:
            spans.append((phase, start / C.RECORD_HZ, (start + count) / C.RECORD_HZ))
            start += count

    for j, name in enumerate(C.JOINT_NAMES):
        ax = axes[j]
        for phase, t0, t1 in spans:
            ax.axvspan(t0, t1, color=PHASE_COLORS[phase], alpha=0.13, lw=0)
        ax.plot(t, states[:, j], lw=1.8, color="#222", label="observation.state")
        ax.plot(t, actions[:, j], lw=1.1, color="#C00", ls="--", label="action (commanded)")
        ax.set_ylabel(name, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)

    for phase, t0, t1 in spans:
        axes[0].text((t0 + t1) / 2, axes[0].get_ylim()[1], phase, ha="center",
                     va="bottom", fontsize=8, color=PHASE_COLORS[phase])

    axes[-1].set_xlabel(f"time (s) @ {C.RECORD_HZ:g} fps")
    fig.suptitle(
        f"SO-101 scripted expert -- seed {meta['seed']} -- "
        f"success={meta['success']} -- grasp={meta['grasp_mechanism']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "trajectories.png", dpi=130)
    print(f"plot     : trajectories.png")
    print(f"\nall artefacts -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
