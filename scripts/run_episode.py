#!/usr/bin/env python
"""Record exactly one scripted-expert episode into LeRobot v0.5.0 format.

    MUJOCO_GL=egl uv run python scripts/run_episode.py --seed 20260807

Everything about the episode -- object pose, phase durations, success -- follows
from the seed alone. Reproducing it means re-running with the same --seed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sim import config as C  # noqa: E402
from sim.episode import run_episode  # noqa: E402
from sim.recorder import write_episode  # noqa: E402

DEFAULT_SEED = 20260807


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=None, help="dataset root (default outputs/episodes/seed_<seed>)")
    ap.add_argument("--repo-id", default="local/so101_sim_pickplace")
    args = ap.parse_args()

    root = pathlib.Path(args.out or C.REPO / "outputs" / "episodes" / f"seed_{args.seed}")

    print(f"seed          : {args.seed}")
    print(f"clocks (Hz)   : physics {C.PHYSICS_HZ} / control {C.CONTROL_HZ} / record {C.RECORD_HZ}")
    print(f"frame         : {C.FRAME_HEIGHT}x{C.FRAME_WIDTH}, cameras {sorted(C.CAMERAS)}")
    print(f"dataset root  : {root}")
    print()

    result = run_episode(seed=args.seed)

    print(f"success       : {result.success}")
    print(f"failure       : {result.failure_reason}")
    print(f"phase steps   : {result.phase_steps}")
    print(f"frames        : {len(result.frames)}  ({len(result.frames) / C.RECORD_HZ:.2f} s)")
    print(f"wall clock    : {result.wall_clock_s:.1f} s")
    print()

    if not result.frames:
        print("no frames recorded; refusing to write an empty dataset")
        return 1

    meta = write_episode(result, root=root, repo_id=args.repo_id)
    print(f"wrote dataset -> {root}")
    print(json.dumps({k: meta[k] for k in ("seed", "success", "failure_reason",
                                           "phase_steps", "n_recorded_frames",
                                           "grasp_mechanism")}, indent=2))
    # A failed episode is still written -- a known-bad episode is useful, a
    # silently-dropped one is not -- but the exit code reports the truth.
    return 0 if result.success else 2


if __name__ == "__main__":
    sys.exit(main())
