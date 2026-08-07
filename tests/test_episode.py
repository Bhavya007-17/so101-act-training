"""Tests for the failure modes that would silently poison downstream training.

The lockstep test is the important one. An observation/action off-by-one does not
crash anything and does not look wrong in a contact sheet -- it surfaces months
later as an unexplained ACT failure. So it is tested by a property that actually
inverts under a shift, not by eyeballing shapes.
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from sim import config as C
from sim.episode import PHASES, _Phase, run_episode
from sim.ik import DiffIK
from sim.recorder import build_features
from sim.scene import Scene

SEED = 20260807


@pytest.fixture(scope="module")
def episode():
    return run_episode(seed=SEED)


# --- the lockstep contract ---------------------------------------------------


def test_action_is_exactly_recomputable_from_the_recorded_observation(episode):
    """The deterministic lockstep check.

    `DiffIK.step` re-seeds from the configuration it is handed, so it is a pure
    function of (q, target). If the action stored on frame k had been computed
    from any state other than the observation stored on frame k -- the previous
    tick's, the next tick's, or a post-step one -- this recomputation would not
    reproduce it. Statistical alignment tests cannot distinguish those cases
    because the IK target always leads the state; this does, exactly.
    """
    ik = DiffIK()
    arm = len(C.ARM_JOINTS)
    for i, f in enumerate(episode.frames):
        expected = ik.step(f.qpos, f.target)[:arm]
        assert np.allclose(expected, f.action[:arm], atol=1e-9), (
            f"frame {i} ({f.phase}): recorded action does not match the action "
            f"implied by its own observation -- observation/action are mispaired"
        )


def test_images_are_rendered_at_the_recorded_timestamp(monkeypatch):
    """Images must come from the same mjData instant as the state, pre-step."""
    seen: list[float] = []
    original = Scene.render_all

    def spy(self):
        seen.append(float(self.data.time))
        return original(self)

    monkeypatch.setattr(Scene, "render_all", spy)
    result = run_episode(seed=SEED)

    recorded = [f.timestamp for f in result.frames]
    assert seen == pytest.approx(recorded, abs=1e-12), (
        "cameras were not rendered at the instants recorded on the frames"
    )


def test_frames_are_on_an_exact_fixed_grid(episode):
    ts = np.array([f.timestamp for f in episode.frames])
    dt = np.diff(ts)
    assert ts[0] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(dt, C.RECORD_DT, atol=1e-9), (
        f"recording grid drifted: dt range [{dt.min()}, {dt.max()}], expected {C.RECORD_DT}"
    )


def test_every_frame_carries_both_cameras_at_the_declared_shape(episode):
    for i, f in enumerate(episode.frames):
        assert set(f.images) == set(C.CAMERAS), f"frame {i} camera set mismatch"
        for key, img in f.images.items():
            assert img.shape == (C.FRAME_HEIGHT, C.FRAME_WIDTH, 3), f"frame {i}/{key}"
            assert img.max() > 0, f"frame {i}/{key} is entirely black"


def test_clock_decimations_are_integers():
    assert C.PHYSICS_HZ / C.CONTROL_HZ == C.PHYSICS_PER_CONTROL
    assert C.CONTROL_HZ / C.RECORD_HZ == C.CONTROL_PER_RECORD


# --- the weld grasp ----------------------------------------------------------


def test_attaching_the_weld_does_not_move_the_object():
    scene = Scene()
    try:
        scene.reset()
        scene.set_object_xy(0.22, 0.03)
        before = scene.object_pos().copy()
        scene.attach()
        scene.step(1)
        moved = float(np.linalg.norm(scene.object_pos() - before))
        assert moved < 1e-4, f"weld teleported the object by {moved:.6f} m"
    finally:
        scene.close()


def test_weld_binds_the_intended_bodies():
    scene = Scene()
    try:
        m = scene.model
        assert m.eq_obj1id[scene.weld_id] == scene.gripper_body
        assert m.eq_obj2id[scene.weld_id] == scene.object_body
        assert not m.eq_active0[scene.weld_id], "weld must start disabled"
    finally:
        scene.close()


# --- phase termination -------------------------------------------------------


def test_phase_times_out_instead_of_hanging():
    """A phase whose waypoint is never reached must trip its budget."""
    unreachable = np.array([10.0, 10.0, 10.0])
    phase = _Phase("approach", [(None, unreachable)], C.GRIPPER_OPEN, budget=5)
    for _ in range(5):
        assert not phase.advance(np.zeros(3))
        phase.steps += 1
    assert phase.timed_out


def test_phase_terminates_on_measured_error_not_step_count():
    target = np.array([0.2, 0.0, 0.05])
    phase = _Phase("approach", [(None, target)], C.GRIPPER_OPEN, budget=1000)
    assert not phase.advance(target + 0.5)          # far away -> keep going
    assert phase.advance(target + C.POS_TOL / 4)    # inside tolerance -> done
    assert not phase.timed_out


def test_all_phases_ran_and_are_reported(episode):
    if episode.success:
        assert set(episode.phase_steps) == set(PHASES)
        assert all(v > 0 for v in episode.phase_steps.values())
        for name, steps in episode.phase_steps.items():
            assert steps < C.PHASE_BUDGET[name], f"{name} hit its budget"


# --- reproducibility and schema ---------------------------------------------


def test_the_seed_alone_reproduces_the_object_pose():
    ik = DiffIK()
    a = run_episode(seed=SEED, ik=ik)
    b = run_episode(seed=SEED, ik=ik)
    assert a.object_pose == pytest.approx(b.object_pose, abs=1e-12)
    assert a.phase_steps == b.phase_steps


def test_sampled_object_pose_is_inside_the_verified_region(episode):
    x, y = episode.object_pose[:2]
    r = float(np.hypot(x, y))
    assert C.SAMPLE_RADIUS_MIN <= r <= C.SAMPLE_RADIUS_MAX
    assert C.SAMPLE_ANGLE_MIN <= float(np.arctan2(y, x)) <= C.SAMPLE_ANGLE_MAX
    assert episode.reachability["fraction_reachable"] > 0.9


def test_feature_schema_matches_the_so101_lerobot_convention():
    f = build_features()
    for key in ("action", "observation.state"):
        assert f[key]["dtype"] == "float32"
        assert f[key]["shape"] == (6,)
        assert f[key]["names"] == [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ]
    for cam in C.CAMERAS:
        img = f[f"observation.images.{cam}"]
        assert img["dtype"] == "video"
        assert img["shape"] == (480, 640, 3)
