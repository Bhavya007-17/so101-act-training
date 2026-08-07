"""The episode runner: four-phase state machine over three decoupled clocks.

THE LOCKSTEP CONTRACT
---------------------
Each recording tick does, in this exact order:

    1. read joint state from mjData                 -> observation at t
    2. render both cameras from that same mjData    -> images at t
    3. compute the action from (1)                  -> action at t
    4. append frame {observation@t, action@t, t}
    5. write the action to the actuators and step physics

Observation and images therefore come from one identical mjData snapshot, and
the recorded action is the one computed *from* that snapshot and applied *after*
it. That is the ACT convention: "given this observation, output this action."
Nothing is read after stepping, so an off-by-one is structurally impossible
rather than merely unlikely. tests/test_episode.py pins this.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from . import config as C
from .ik import DiffIK
from .scene import Scene
from .workspace import Sample, grasp_targets, place_targets, sample_object_xy, survey

PHASES = ("approach", "grasp", "transport", "release")


@dataclasses.dataclass
class Frame:
    """One recording tick. Every field is captured from the same instant."""

    timestamp: float
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray
    images: dict[str, np.ndarray]
    phase: str
    # The IK target this action was solved against. Not written to the dataset;
    # kept so tests can recompute the action from the recorded observation and
    # prove the two were produced at the same instant.
    target: object = None


@dataclasses.dataclass
class EpisodeResult:
    frames: list[Frame]
    success: bool
    failure_reason: str | None
    phase_steps: dict[str, int]
    seed: int
    object_pose: list[float]
    target_pose: list[float]
    sample: Sample
    reachability: dict
    wall_clock_s: float
    final_object_pos: list[float]
    place_error_m: float


class _Phase:
    """A waypoint sequence with a gripper command and a step budget."""

    def __init__(self, name, waypoints, gripper, budget, settle=0):
        self.name = name
        self.waypoints = waypoints          # list[(SE3 target, position ndarray)]
        self.gripper = gripper
        self.budget = budget
        self.settle = settle
        self.index = 0
        self.steps = 0
        self.settled = 0

    @property
    def target(self):
        return self.waypoints[self.index]

    def advance(self, ee_pos: np.ndarray) -> bool:
        """Consume waypoints as they are reached. True once the phase is done.

        Termination is a measured Cartesian error, never a step count. The budget
        exists only so a phase that cannot converge fails instead of hanging.
        """
        _, wp_pos = self.target
        if float(np.linalg.norm(ee_pos - wp_pos)) < C.POS_TOL:
            if self.index < len(self.waypoints) - 1:
                self.index += 1
                return False
            self.settled += 1
            return self.settled >= self.settle
        return False

    @property
    def timed_out(self) -> bool:
        return self.steps >= self.budget


def _wp(ik: DiffIK, se3):
    return (se3, np.asarray(se3.translation()))


def run_episode(seed: int, scene: Scene | None = None, ik: DiffIK | None = None) -> EpisodeResult:
    """Run one pick-and-place episode end to end and return everything recorded."""
    t_start = time.perf_counter()
    owns_scene = scene is None
    scene = scene or Scene()
    ik = ik or DiffIK()

    rng = np.random.default_rng(seed)

    scene.reset()
    q_home = scene.qpos[: len(C.ARM_JOINTS) + 1].copy()

    # Reachability is measured before anything moves, and the sampled object pose
    # is rejection-tested against the same oracle.
    reachability = survey(ik, q_home)
    sample = sample_object_xy(rng, ik, q_home)

    scene.reset()
    scene.set_object_xy(*sample.xy)
    object_pose = scene.object_pos().tolist()
    target_pose = scene.target_pos().tolist()

    # The scene's target_zone body is authoritative for where the object must go,
    # so the green marker and the commanded place pose can never disagree.
    target_xy = tuple(scene.target_pos()[:2])

    pregrasp, grasp = grasp_targets(ik, sample.xy)
    lift_pos = np.array([sample.xy[0], sample.xy[1], C.LIFT_HEIGHT])
    lift = ik.target(lift_pos, ik.grasp_quat(sample.xy))
    above_target, release_pose = place_targets(ik, target_xy)

    phases = [
        _Phase("approach", [_wp(ik, pregrasp), _wp(ik, grasp)], C.GRIPPER_OPEN,
               C.PHASE_BUDGET["approach"]),
        _Phase("grasp", [_wp(ik, grasp)], C.GRIPPER_CLOSED,
               C.PHASE_BUDGET["grasp"], settle=C.GRASP_SETTLE_STEPS),
        _Phase("transport", [_wp(ik, lift), _wp(ik, above_target), _wp(ik, release_pose)],
               C.GRIPPER_CLOSED, C.PHASE_BUDGET["transport"]),
        _Phase("release", [_wp(ik, release_pose)], C.GRIPPER_OPEN,
               C.PHASE_BUDGET["release"], settle=C.RELEASE_SETTLE_STEPS),
    ]

    frames: list[Frame] = []
    phase_steps = {p: 0 for p in PHASES}
    failure_reason: str | None = None
    control_tick = 0

    for phase in phases:
        # The weld is scripted at phase boundaries, not inferred from contact.
        if phase.name == "transport" and not scene.attached:
            scene.attach()
        if phase.name == "release" and scene.attached:
            scene.release()

        while True:
            # --- 1. observation, from the current mjData -----------------------
            qpos = scene.qpos
            qvel = scene.qvel
            timestamp = float(scene.data.time)
            is_record_tick = control_tick % C.CONTROL_PER_RECORD == 0

            # --- 2. images, from that same mjData -----------------------------
            images = scene.render_all() if is_record_tick else None

            # --- 3. action, computed from that observation --------------------
            se3, _ = phase.target
            q_cmd = ik.step(qpos, se3)
            action = np.concatenate([q_cmd[: len(C.ARM_JOINTS)], [phase.gripper]])

            # --- 4. record the pair -------------------------------------------
            if is_record_tick:
                frames.append(
                    Frame(
                        timestamp=timestamp,
                        qpos=qpos,
                        qvel=qvel,
                        action=action.astype(np.float32),
                        images=images,
                        phase=phase.name,
                        target=se3,
                    )
                )

            # --- 5. apply, then advance physics -------------------------------
            scene.set_ctrl(action)
            scene.step(C.PHYSICS_PER_CONTROL)

            phase.steps += 1
            control_tick += 1

            if phase.advance(scene.ee_pose().pos):
                break
            if phase.timed_out:
                failure_reason = f"{phase.name}_timeout"
                break

        phase_steps[phase.name] = phase.steps
        if failure_reason:
            break

    if scene.attached:
        scene.release()
        scene.step(C.PHYSICS_PER_CONTROL * 20)  # let the object settle after release

    final_object = scene.object_pos()
    place_error = float(np.linalg.norm(final_object[:2] - np.asarray(target_xy)))
    in_zone = place_error < C.TARGET_ZONE_RADIUS and final_object[2] < 0.06

    if failure_reason is None and not in_zone:
        failure_reason = (
            f"object_not_in_target_zone (xy error {place_error:.3f} m, z {final_object[2]:.3f} m)"
        )

    result = EpisodeResult(
        frames=frames,
        success=failure_reason is None,
        failure_reason=failure_reason,
        phase_steps=phase_steps,
        seed=seed,
        object_pose=object_pose,
        target_pose=target_pose,
        sample=sample,
        reachability=reachability,
        wall_clock_s=time.perf_counter() - t_start,
        final_object_pos=final_object.tolist(),
        place_error_m=place_error,
    )
    if owns_scene:
        scene.close()
    return result


def metadata(result: EpisodeResult) -> dict:
    """Everything needed to reproduce and to interpret the episode."""
    survey_summary = {
        k: v for k, v in result.reachability.items() if k != "grid"
    }
    return {
        "seed": result.seed,
        "success": result.success,
        "failure_reason": result.failure_reason,
        "phase_steps": result.phase_steps,
        "phase_steps_unit": "control ticks at %g Hz" % C.CONTROL_HZ,
        "frames_per_phase": {
            p: sum(1 for f in result.frames if f.phase == p) for p in PHASES
        },
        "n_recorded_frames": len(result.frames),
        "wall_clock_s": round(result.wall_clock_s, 2),
        "sampled_object_pose": result.object_pose,
        "target_pose": result.target_pose,
        "final_object_pos": result.final_object_pos,
        "place_error_m": round(result.place_error_m, 4),
        "sample": {
            "xy": list(result.sample.xy),
            "radius_m": round(result.sample.radius, 4),
            "azimuth_rad": round(result.sample.azimuth, 4),
            "rejection_attempts": result.sample.attempts,
            "pregrasp_ik_error_m": round(result.sample.pregrasp_error, 5),
            "grasp_ik_error_m": round(result.sample.grasp_error, 5),
        },
        "reachability": survey_summary,
        "clocks_hz": {
            "physics": C.PHYSICS_HZ,
            "control": C.CONTROL_HZ,
            "recording": C.RECORD_HZ,
        },
        "decimation": {
            "physics_per_control": C.PHYSICS_PER_CONTROL,
            "control_per_record": C.CONTROL_PER_RECORD,
        },
        "action_space": {
            "representation": "absolute_joint_position_targets",
            "units": "radians",
            "names": C.FEATURE_NAMES,
            "note": (
                "Actuator ctrl values, i.e. absolute joint-position setpoints. "
                "Sim records radians; SO-101 hardware reports normalised .pos units, "
                "so deployment needs one affine per-joint rescale and no change of "
                "representation."
            ),
        },
        "grasp_mechanism": "weld_constraint",
        "grasp_mechanism_note": (
            "Grasp is scripted: a MuJoCo equality weld between the gripper and the "
            "object is toggled at a phase boundary via mjData.eq_active. Grasp "
            "success is therefore imposed, not physically simulated, so contact "
            "precision and grasp robustness are NOT measurable from this data."
        ),
        "model_provenance": {
            "source": "google-deepmind/mujoco_menagerie",
            "model": C.MENAGERIE_MODEL,
            "commit": C.MENAGERIE_COMMIT,
        },
    }
