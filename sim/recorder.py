"""Write an episode through the installed LeRobotDataset 0.5.0 API.

The feature schema deliberately mirrors lerobot/svla_so101_pickplace -- the
dataset this repo already trained ACT on -- so ACT consumes it with no custom
adapter: float32 [6] `action` and `observation.state` carrying the same six
`<joint>.pos` names, plus `observation.images.*` video streams at 480x640.

API surface used, read from the installed source rather than assumed:
  LeRobotDataset.create(repo_id, fps, features, root=, robot_type=, use_videos=)
  .add_frame(frame_dict)      # expects a `task` key alongside the features
  .save_episode()
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from . import config as C
from .episode import EpisodeResult, metadata


def build_features() -> dict:
    """The ACT-recognisable feature schema."""
    features = {
        "action": {
            "dtype": "float32",
            "shape": (C.N_JOINTS,),
            "names": C.FEATURE_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (C.N_JOINTS,),
            "names": C.FEATURE_NAMES,
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (C.N_JOINTS,),
            "names": C.FEATURE_NAMES,
        },
    }
    for key in C.CAMERAS:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (C.FRAME_HEIGHT, C.FRAME_WIDTH, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def write_episode(
    result: EpisodeResult,
    root: pathlib.Path,
    repo_id: str = "local/so101_sim_pickplace",
) -> dict:
    """Write one episode to `root` in LeRobot v3.0 format. Returns the metadata."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = pathlib.Path(root)
    if root.exists():
        raise FileExistsError(
            f"{root} already exists; refusing to write over an existing dataset"
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(C.RECORD_HZ),
        features=build_features(),
        root=root,
        robot_type=C.ROBOT_TYPE,
        use_videos=True,
    )

    for frame in result.frames:
        payload = {
            "action": frame.action.astype(np.float32),
            "observation.state": frame.qpos.astype(np.float32),
            "observation.velocity": frame.qvel.astype(np.float32),
            "task": C.TASK_DESCRIPTION,
        }
        for key, img in frame.images.items():
            payload[f"observation.images.{key}"] = img
        dataset.add_frame(payload)

    dataset.save_episode()

    meta = metadata(result)
    meta["dataset"] = {
        "repo_id": repo_id,
        "root": str(root),
        "fps": int(C.RECORD_HZ),
        "frame_shape": [C.FRAME_HEIGHT, C.FRAME_WIDTH, 3],
        "features": sorted(build_features()),
        "codebase_version": "v3.0",
    }

    # The per-episode record lives beside the dataset, not only in the write-up,
    # so the weld limitation travels with the data.
    (root / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
    (root / "reachability_survey.json").write_text(
        json.dumps(result.reachability, indent=2)
    )
    return meta
