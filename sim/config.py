"""Single source of truth for the episode runner's constants.

Every number that governs timing, geometry or termination lives here so the
episode metadata can quote it and EXPERT.md can't drift out of sync with the code.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
SCENE_XML = REPO / "sim" / "assets" / "so101" / "scene_pickplace.xml"
ROBOT_XML = REPO / "sim" / "assets" / "so101" / "so101.xml"

# Provenance of the vendored robot model.
MENAGERIE_COMMIT = "c1a4eeb85694ae1dffe33ff1797d4e528928a133"
MENAGERIE_MODEL = "robotstudio_so101"

# --- The three clocks -------------------------------------------------------
# They must divide evenly, or recording ticks drift against physics steps and
# the observation/action pairing stops being exact. 240/60/30 gives integer
# decimations of 4 and 8.
PHYSICS_HZ = 240.0            # mjModel.opt.timestep = 1/240 (set in the scene XML)
CONTROL_HZ = 60.0             # IK solve + ctrl write
RECORD_HZ = 30.0              # dataset frame capture; matches a real 30 fps camera

PHYSICS_DT = 1.0 / PHYSICS_HZ
CONTROL_DT = 1.0 / CONTROL_HZ
RECORD_DT = 1.0 / RECORD_HZ

PHYSICS_PER_CONTROL = round(PHYSICS_HZ / CONTROL_HZ)   # 4
CONTROL_PER_RECORD = round(CONTROL_HZ / RECORD_HZ)     # 2

# --- Cameras ----------------------------------------------------------------
# 480x640 is deliberate: it is exactly the geometry of lerobot/svla_so101_pickplace,
# which this repo already trained ACT on, so the measured batch-size ladder in
# DECISIONS.md (bs=12 -> 71% of 8 GB with two streams) stays valid.
FRAME_HEIGHT = 480
FRAME_WIDTH = 640
CAMERAS = {
    "external": "external",     # fixed 3/4 view of the workspace
    "wrist": "wrist_cam",       # menagerie's own wrist-mounted camera
}

# --- Joints -----------------------------------------------------------------
# Order is the model's joint order AND the LeRobot SO-101 feature order. They
# coincide by construction, which is why no adapter is needed.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ARM_JOINTS = JOINT_NAMES[:5]      # driven by IK
GRIPPER_JOINT = JOINT_NAMES[5]    # driven by the state machine
N_JOINTS = len(JOINT_NAMES)
FEATURE_NAMES = [f"{j}.pos" for j in JOINT_NAMES]

# Gripper actuator ctrl endpoints (radians; jnt range is -0.174533..1.745329).
GRIPPER_OPEN = 1.2
GRIPPER_CLOSED = 0.05

EE_SITE = "gripperframe"
OBJECT_BODY = "object"
GRIPPER_BODY = "gripper"
WELD_NAME = "grasp_weld"
TARGET_BODY = "target_zone"
HOME_KEYFRAME = "home"

# --- IK ---------------------------------------------------------------------
IK_SOLVER = "daqp"
IK_DAMPING = 1e-2
IK_POS_COST = 1.0
IK_POSTURE_COST = 1e-3
IK_ITERS_PER_CONTROL = 1       # one solve per control tick: the standard diff-IK loop

# Task gain and joint speed ceiling. Without these the QP drives the whole
# Cartesian error inside a single tick and the arm effectively teleports -- the
# first working episode ran 1.4 s end to end, which is useless as demonstration
# data and impossible on real STS3215 servos. Capping joint speed makes the
# trajectory both watchable and hardware-plausible.
IK_TASK_GAIN = 0.4
MAX_JOINT_VEL = 0.7            # rad/s, applied to every arm joint

# Per-axis orientation cost, in the end-effector's LOCAL frame.
#
# The arm has five actuated DOF available to the end-effector pose (the sixth
# joint is the gripper), so a full 6-DOF pose target is over-constrained and
# measurably unreachable above z ~= 0.05. The gripper's local x-axis is its
# approach direction, so zeroing the first component frees roll *about* the
# approach axis -- the one rotation a symmetric top-down grasp does not care
# about -- while pinning the approach direction itself.
#
# Measured: this holds the approach axis at dot(-z_world) = 0.997-0.998 across
# the whole sampling region. The alternative of freeing local z instead reaches
# further but lets the gripper tilt to dot = 0.545, i.e. ~57 degrees off
# vertical, which is not a top-down grasp at all.
IK_ORI_COST = [0.0, 0.5, 0.5]

# --- Phase termination ------------------------------------------------------
# Every phase ends on a measured pose error, never on a step count. The budget
# exists only to trip a timeout so a non-converging phase fails instead of hanging.
POS_TOL = 0.012                # m, end-effector position error
GRASP_SETTLE_STEPS = 12        # control ticks held closed before the weld engages
# The jaw is a real actuated DOF with its own first-order response, so a short
# release phase ends the episode while the gripper is still opening. 36 ticks
# (0.6 s) lets the commanded open actually be reached before the last frame.
RELEASE_SETTLE_STEPS = 36      # control ticks held open after the weld releases
PHASE_BUDGET = {               # control ticks
    "approach": 400,
    "grasp": 120,
    "transport": 500,
    "release": 120,
}

# Cartesian offsets used to build phase targets.
#
# All of these sit inside the measured top-down envelope: with the approach axis
# pinned, IK converges for z in [0.02, 0.065] and radius in [0.16, 0.28]. The
# arm is short, so the whole manoeuvre is deliberately shallow. Asking for the
# 0.14 m transport height that looks natural on a bigger arm puts the target
# outside the reachable set and every episode times out in approach.
PREGRASP_HEIGHT = 0.035        # m above the grasp pose before descending
GRASP_HEIGHT = 0.005           # m: EE site z offset above object centre
LIFT_HEIGHT = 0.055            # m, absolute EE z during transport
PLACE_HEIGHT = 0.045           # m, absolute EE z at release

# --- Randomisation ----------------------------------------------------------
# Sampled in polar coordinates about the arm base, then rejection-tested for
# actual IK reachability. Bounds are pulled inside the measured envelope so the
# rejection rate stays low; sim/workspace.py:survey re-measures it every run.
SAMPLE_RADIUS_MIN = 0.17
SAMPLE_RADIUS_MAX = 0.27
SAMPLE_ANGLE_MIN = -0.60       # rad about world +x
SAMPLE_ANGLE_MAX = 0.60
OBJECT_HALF_SIZE = 0.015
# NOTE: the authoritative target position is the `target_zone` body in the scene
# XML -- read it with Scene.target_pos(). This constant exists only so offline
# reachability probes can run without instantiating a Scene, and the scene XML
# must be kept in step with it. Never place against this value at runtime.
TARGET_XY = (0.10, 0.22)
TARGET_ZONE_RADIUS = 0.04

TASK_DESCRIPTION = "Pick up the red cube and place it in the green target zone."
ROBOT_TYPE = "so101_follower_sim"
