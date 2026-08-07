"""Scene wrapper: model/data ownership, camera rendering, and the weld grasp.

Deliberately thin. It owns MuJoCo state and nothing about the task, so the
episode state machine can be read without also reading MuJoCo boilerplate.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from . import config as C


@dataclasses.dataclass(frozen=True)
class Pose:
    """A rigid pose. Frozen: poses are values, never mutated in place."""

    pos: np.ndarray
    quat: np.ndarray  # (w, x, y, z)

    @staticmethod
    def from_site(data: mujoco.MjData, site_id: int) -> "Pose":
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        return Pose(pos=data.site_xpos[site_id].copy(), quat=quat)


class Scene:
    """Owns the MuJoCo model/data pair and everything that touches them."""

    def __init__(self, xml_path=None, height=C.FRAME_HEIGHT, width=C.FRAME_WIDTH):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or C.SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)

        n = mujoco.mj_name2id
        self.joint_ids = [n(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in C.JOINT_NAMES]
        self.qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.actuator_ids = [
            n(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in C.JOINT_NAMES
        ]
        self.ee_site = n(self.model, mujoco.mjtObj.mjOBJ_SITE, C.EE_SITE)
        self.object_body = n(self.model, mujoco.mjtObj.mjOBJ_BODY, C.OBJECT_BODY)
        self.gripper_body = n(self.model, mujoco.mjtObj.mjOBJ_BODY, C.GRIPPER_BODY)
        self.target_body = n(self.model, mujoco.mjtObj.mjOBJ_BODY, C.TARGET_BODY)
        self.weld_id = n(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, C.WELD_NAME)
        self.object_qpos_adr = self.model.jnt_qposadr[
            self.model.body_jntadr[self.object_body]
        ]

        missing = {
            k: v
            for k, v in {
                "ee_site": self.ee_site,
                "object_body": self.object_body,
                "gripper_body": self.gripper_body,
                "weld": self.weld_id,
            }.items()
            if v < 0
        }
        if missing:
            raise RuntimeError(f"scene is missing required elements: {sorted(missing)}")

        self.cam_ids = {}
        for key, cam_name in C.CAMERAS.items():
            cid = n(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cid < 0:
                raise RuntimeError(f"camera {cam_name!r} not found in model")
            self.cam_ids[key] = cid

    # --- state ---------------------------------------------------------------

    def reset(self) -> None:
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, C.HOME_KEYFRAME)
        if key < 0:
            mujoco.mj_resetData(self.model, self.data)
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        self.release()
        mujoco.mj_forward(self.model, self.data)

    @property
    def qpos(self) -> np.ndarray:
        """The 6 robot joint positions, in LeRobot feature order."""
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    @property
    def ctrl(self) -> np.ndarray:
        return self.data.ctrl[self.actuator_ids].copy()

    def set_ctrl(self, targets: np.ndarray) -> None:
        """Write joint-position targets, clipped to each actuator's ctrlrange."""
        lo = self.model.actuator_ctrlrange[self.actuator_ids, 0]
        hi = self.model.actuator_ctrlrange[self.actuator_ids, 1]
        self.data.ctrl[self.actuator_ids] = np.clip(targets, lo, hi)

    def set_object_xy(self, x: float, y: float, z: float | None = None) -> None:
        a = self.object_qpos_adr
        self.data.qpos[a : a + 3] = [x, y, C.OBJECT_HALF_SIZE if z is None else z]
        self.data.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[
            self.model.jnt_dofadr[self.model.body_jntadr[self.object_body]] :
        ][:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # --- poses ---------------------------------------------------------------

    def ee_pose(self) -> Pose:
        return Pose.from_site(self.data, self.ee_site)

    def object_pos(self) -> np.ndarray:
        return self.data.xpos[self.object_body].copy()

    def target_pos(self) -> np.ndarray:
        return self.data.xpos[self.target_body].copy()

    # --- the weld grasp ------------------------------------------------------

    def attach(self) -> None:
        """Engage the weld, re-anchored to the live relative pose.

        Anchoring matters: if eq_data kept its compile-time relpose the object
        would teleport to wherever it sat at model-build time the instant the
        constraint went active. Recomputing it here makes attachment a no-op on
        the object's pose, which is what a grasp should look like.

        Grasp success is scripted, not physical. Nothing here reads contact
        forces, so this data cannot measure grasp precision.
        """
        rel_pos, rel_quat = self._relative_pose()
        eq = self.model.eq_data[self.weld_id]
        eq[0:3] = 0.0            # anchor at body2 origin
        eq[3:6] = rel_pos        # relpose position
        eq[6:10] = rel_quat      # relpose quaternion (w, x, y, z)
        eq[10] = 1.0             # torquescale
        self.data.eq_active[self.weld_id] = 1
        mujoco.mj_forward(self.model, self.data)

    def release(self) -> None:
        self.data.eq_active[self.weld_id] = 0

    @property
    def attached(self) -> bool:
        return bool(self.data.eq_active[self.weld_id])

    def _relative_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Pose of the object body expressed in the gripper body's frame."""
        p1 = self.data.xpos[self.gripper_body]
        q1 = np.empty(4)
        mujoco.mju_mat2Quat(q1, self.data.xmat[self.gripper_body])
        p2 = self.data.xpos[self.object_body]
        q2 = np.empty(4)
        mujoco.mju_mat2Quat(q2, self.data.xmat[self.object_body])

        q1_inv = np.empty(4)
        mujoco.mju_negQuat(q1_inv, q1)

        d = p2 - p1
        rel_pos = np.empty(3)
        mujoco.mju_rotVecQuat(rel_pos, d, q1_inv)
        rel_quat = np.empty(4)
        mujoco.mju_mulQuat(rel_quat, q1_inv, q2)
        return rel_pos, rel_quat

    # --- rendering -----------------------------------------------------------

    def render(self, camera_key: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self.cam_ids[camera_key])
        return self._renderer.render()

    def render_all(self) -> dict[str, np.ndarray]:
        return {k: self.render(k) for k in self.cam_ids}

    def close(self) -> None:
        # Idempotent: the Renderer's own __del__ also calls close(), and freeing
        # an EGL context twice raises a confusing EGLError at interpreter exit.
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
