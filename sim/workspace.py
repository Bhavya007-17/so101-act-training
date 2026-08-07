"""Object-pose randomisation, gated on measured IK reachability.

The sampling region is polar about the arm base, but a polar annulus is only a
*prior*. Nothing here assumes it is safe: every drawn sample is rejected unless
the IK solver actually converges to both the pregrasp and the grasp pose for it,
and `survey` characterises the whole region up front so the acceptance rate is a
measured number rather than a hope.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import config as C
from .ik import DiffIK


@dataclasses.dataclass(frozen=True)
class Sample:
    xy: tuple[float, float]
    radius: float
    azimuth: float
    attempts: int
    pregrasp_error: float
    grasp_error: float


def grasp_targets(ik: DiffIK, xy, object_z: float = C.OBJECT_HALF_SIZE):
    """(pregrasp, grasp) SE3 targets for an object standing at `xy`."""
    quat = ik.grasp_quat(xy)
    grasp_pos = np.array([xy[0], xy[1], object_z + C.GRASP_HEIGHT])
    pregrasp_pos = grasp_pos + np.array([0.0, 0.0, C.PREGRASP_HEIGHT])
    return ik.target(pregrasp_pos, quat), ik.target(grasp_pos, quat)


def place_targets(ik: DiffIK, xy=C.TARGET_XY):
    """(above-target, release) SE3 targets for the drop zone."""
    quat = ik.grasp_quat(xy)
    release_pos = np.array([xy[0], xy[1], C.PLACE_HEIGHT])
    above_pos = np.array([xy[0], xy[1], C.LIFT_HEIGHT])
    return ik.target(above_pos, quat), ik.target(release_pos, quat)


def is_reachable(ik: DiffIK, xy, q_home: np.ndarray) -> tuple[bool, float, float]:
    """Reachable == IK converges to pregrasp *and* to grasp, within joint limits."""
    pregrasp, grasp = grasp_targets(ik, xy)
    q1, e1, ok1 = ik.solve_to(pregrasp, q_home)
    if not ok1:
        return False, e1, np.inf
    _, e2, ok2 = ik.solve_to(grasp, q1)
    return bool(ok1 and ok2), e1, e2


def sample_object_xy(
    rng: np.random.Generator,
    ik: DiffIK,
    q_home: np.ndarray,
    max_attempts: int = 200,
) -> Sample:
    """Rejection-sample an object position that is provably IK-reachable.

    Radius is sampled as sqrt(U) between the bounds so points are uniform over
    the annulus *area* rather than clustering near the inner edge.
    """
    for attempt in range(1, max_attempts + 1):
        u = rng.uniform()
        radius = float(
            np.sqrt(
                C.SAMPLE_RADIUS_MIN**2
                + u * (C.SAMPLE_RADIUS_MAX**2 - C.SAMPLE_RADIUS_MIN**2)
            )
        )
        azimuth = float(rng.uniform(C.SAMPLE_ANGLE_MIN, C.SAMPLE_ANGLE_MAX))
        xy = (radius * np.cos(azimuth), radius * np.sin(azimuth))

        ok, e1, e2 = is_reachable(ik, xy, q_home)
        if ok:
            return Sample(
                xy=(float(xy[0]), float(xy[1])),
                radius=radius,
                azimuth=azimuth,
                attempts=attempt,
                pregrasp_error=float(e1),
                grasp_error=float(e2),
            )
    raise RuntimeError(
        f"no reachable object pose in {max_attempts} attempts -- "
        "the sampling annulus in sim/config.py is wrong for this arm"
    )


def survey(ik: DiffIK, q_home: np.ndarray, n_radius: int = 9, n_angle: int = 9) -> dict:
    """Grid-sweep the sampling region and measure what fraction is reachable.

    Run once and recorded in the episode metadata, so the claim "the sampling
    region is reachable" is backed by a number instead of an assumption.
    """
    radii = np.linspace(C.SAMPLE_RADIUS_MIN, C.SAMPLE_RADIUS_MAX, n_radius)
    angles = np.linspace(C.SAMPLE_ANGLE_MIN, C.SAMPLE_ANGLE_MAX, n_angle)
    grid, reachable = [], 0
    for r in radii:
        for a in angles:
            xy = (float(r * np.cos(a)), float(r * np.sin(a)))
            ok, e1, e2 = is_reachable(ik, xy, q_home)
            reachable += int(ok)
            grid.append(
                {
                    "xy": xy,
                    "radius": float(r),
                    "azimuth": float(a),
                    "reachable": bool(ok),
                    "pregrasp_error": float(e1),
                    "grasp_error": float(e2) if np.isfinite(e2) else None,
                }
            )
    above, release = place_targets(ik)
    _, tgt_err, tgt_ok = ik.solve_to(release, q_home)
    return {
        "n_probed": len(grid),
        "n_reachable": reachable,
        "fraction_reachable": reachable / len(grid),
        "radius_range_m": [C.SAMPLE_RADIUS_MIN, C.SAMPLE_RADIUS_MAX],
        "azimuth_range_rad": [C.SAMPLE_ANGLE_MIN, C.SAMPLE_ANGLE_MAX],
        "target_zone_xy": list(C.TARGET_XY),
        "target_zone_reachable": bool(tgt_ok),
        "target_zone_error_m": float(tgt_err),
        "grid": grid,
    }
