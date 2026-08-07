"""Differential IK via mink, plus the reachability test built on top of it.

mink's Configuration builds its own MjData over whatever model it is handed. The
full scene carries the object's freejoint, so solving there would put six object
DOFs into the QP's decision variables. This module therefore solves on the
robot-only model (nq = nv = 6) and hands the result back as joint-position
targets, which the caller writes into the scene's actuators. IK stays purely
kinematic and never sees the object.
"""

from __future__ import annotations

import mujoco
import mink
import numpy as np

from . import config as C

# The arm configuration from menagerie's own `pickup` keyframe (scene_box.xml).
# Used only to harvest a known-good "gripper is grasping something on the table"
# end-effector orientation, rather than hand-deriving a down-pointing quaternion.
REF_GRASP_QPOS = np.array([0.0, 0.000381818, 0.473496, 1.17717, 1.58437, 0.727663])


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return quat


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class DiffIK:
    """Differential IK on the robot-only model."""

    def __init__(self, robot_xml=None):
        self.model = mujoco.MjModel.from_xml_path(str(robot_xml or C.ROBOT_XML))
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, C.EE_SITE)
        if self.site < 0:
            raise RuntimeError(f"site {C.EE_SITE!r} not in robot model")

        self.configuration = mink.Configuration(self.model)
        self.frame_task = mink.FrameTask(
            frame_name=C.EE_SITE,
            frame_type="site",
            position_cost=C.IK_POS_COST,
            orientation_cost=C.IK_ORI_COST,
            gain=C.IK_TASK_GAIN,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=C.IK_POSTURE_COST)
        self.posture_task.set_target(REF_GRASP_QPOS.copy())
        self.tasks = [self.frame_task, self.posture_task]
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(
                self.model, {j: C.MAX_JOINT_VEL for j in C.JOINT_NAMES}
            ),
        ]

        self._ref_rot, self._ref_azimuth = self._reference_orientation()

    def _reference_orientation(self) -> tuple[np.ndarray, float]:
        """FK the menagerie grasp pose and keep its EE rotation as the template."""
        self.data.qpos[:] = REF_GRASP_QPOS
        mujoco.mj_kinematics(self.model, self.data)
        rot = self.data.site_xmat[self.site].reshape(3, 3).copy()
        pos = self.data.site_xpos[self.site].copy()
        return rot, float(np.arctan2(pos[1], pos[0]))

    def grasp_quat(self, xy) -> np.ndarray:
        """Grasp orientation for a target at world `xy`.

        The reference rotation, yawed about world z by the difference between the
        target's azimuth and the reference azimuth, so the gripper faces the
        object the same way at every point in the workspace.
        """
        azimuth = float(np.arctan2(xy[1], xy[0]))
        return _mat_to_quat(_rot_z(azimuth - self._ref_azimuth) @ self._ref_rot)

    def target(self, pos, quat) -> mink.SE3:
        return mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(quat, dtype=float)), np.asarray(pos, dtype=float)
        )

    # --- online use, at control rate ----------------------------------------

    def step(self, q: np.ndarray, target: mink.SE3, dt: float = C.CONTROL_DT) -> np.ndarray:
        """One control tick: return updated joint-position targets.

        `q` is the arm's measured configuration, so the solver is re-seeded from
        the real robot state every tick and cannot drift away from it.
        """
        self.configuration.update(np.asarray(q, dtype=float).copy())
        self.frame_task.set_target(target)
        for _ in range(C.IK_ITERS_PER_CONTROL):
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt,
                C.IK_SOLVER,
                damping=C.IK_DAMPING,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, dt)
        return self.configuration.q.copy()

    def position_error(self, q: np.ndarray, target_pos: np.ndarray) -> float:
        """Euclidean EE position error at configuration `q`. Drives phase exit."""
        self.data.qpos[:] = np.asarray(q, dtype=float)
        mujoco.mj_kinematics(self.model, self.data)
        return float(np.linalg.norm(self.data.site_xpos[self.site] - np.asarray(target_pos)))

    # --- offline use, for reachability --------------------------------------

    def solve_to(
        self,
        target: mink.SE3,
        q_init: np.ndarray,
        max_iters: int = 800,
        tol: float = C.POS_TOL,
    ) -> tuple[np.ndarray, float, bool]:
        """Run IK to convergence from `q_init`. Returns (q, pos_error, converged).

        This is the reachability oracle: a pose counts as reachable only if the
        solver actually converges to it inside the joint limits.
        """
        self.configuration.update(np.asarray(q_init, dtype=float).copy())
        self.frame_task.set_target(target)
        target_pos = target.translation()
        err = np.inf
        for _ in range(max_iters):
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                C.CONTROL_DT,
                C.IK_SOLVER,
                damping=C.IK_DAMPING,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, C.CONTROL_DT)
            err = float(
                np.linalg.norm(
                    self.configuration.data.site_xpos[self.site] - target_pos
                )
            )
            if err < tol:
                break
        q = self.configuration.q.copy()
        lo = self.model.jnt_range[:, 0]
        hi = self.model.jnt_range[:, 1]
        within = bool(np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6))
        return q, err, bool(err < tol and within)
