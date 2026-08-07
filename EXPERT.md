# EXPERT.md — scripted-expert episode generation

The scripted expert: a mink differential-IK controller driving a MuJoCo SO-101
through approach → weld-grasp → transport → release, logging two camera streams
and joint states at a fixed rate into LeRobot v3.0 format.

**Scope of this document:** one episode, recorded end to end and inspectable. Not
a dataset. Multi-episode generation, ACT training and Hub upload are explicitly
out of scope.

---

## 0. What was already here, and what wasn't

This repo was a LeRobot 0.5.0 **toolchain validation** environment — an ACT
training run on a downloaded HF dataset, five verification scripts, no
simulation code. There was no MuJoCo manipulation setup on the machine: no
`mink`, no MJCF, no IK loop, no weld. The only MuJoCo artefact anywhere was an
unpacked `mujoco-3.9.0` C SDK tarball in a second, stale WSL distro, wired to
nothing.

So the IK controller, the weld grasp and the camera rig documented below were
built here rather than wrapped. **They are therefore not the controller that will
be deployed** unless this one is adopted as such.

The robot model is *not* hand-rolled: it is `robotstudio_so101` vendored from
`google-deepmind/mujoco_menagerie` at commit
`c1a4eeb85694ae1dffe33ff1797d4e528928a133`, into `sim/assets/so101/`, unmodified.
Only `scene_pickplace.xml` beside it is ours.

---

## 1. Decision: action representation → **absolute joint-position targets**

**Chosen:** a 6-vector of absolute joint-position setpoints, in the model's joint
order, recorded as both `action` and `observation.state`.

The decisive evidence is in this repo already. `lerobot/svla_so101_pickplace` —
the dataset whose ACT run validated this toolchain — has `action` and
`observation.state` both `float32 [6]` with names
`shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos`.
The menagerie SO-101 exposes exactly those six joints, in exactly that order,
driven by six MuJoCo `position` actuators. The sim's native command *is* the
hardware's native command. Nothing has to be translated.

**Why not end-effector deltas.** Three costs, all paid at deployment:

1. **An SO-101 follower accepts joint positions, not Cartesian deltas.** An EE-delta
   policy needs an IK solver *in the control loop on hardware* — so the runtime
   acquires a dependency that can fail to converge, and mismatches between the
   deployment IK and the one that generated the data become silent distribution
   shift.
2. **This arm is 5-DOF for pose.** Only five joints move the end-effector (the sixth
   is the jaw), so a 6-DOF Cartesian command is over-constrained and its mapping
   back to joints is not unique. Measured here: with full orientation constrained,
   IK stops converging above z ≈ 0.05 m (§5). A joint-space action never has this
   problem because it *is* the configuration.
3. **Deltas are frame- and rate-dependent.** Their meaning changes with the control
   period and the frame convention; absolute joint targets are invariant to both,
   so a rate change is a resampling problem rather than a relabelling problem.

**The one cost of this choice, stated plainly.** Sim records **radians**; SO-101
hardware reports normalised `.pos` units. That is a per-joint affine rescale —
a change of units, not of representation — and it is recorded in
`episode_metadata.json → action_space`. Calibrate against the real arm before
mixing sim and real episodes in one dataset.

## 2. Decision: recording rate **30 fps**, resolution **480×640**, two cameras

**Chosen:** 30 Hz, 480×640×3 RGB, `external` + `wrist`.

**Why 480×640.** It is exactly the geometry of `svla_so101_pickplace`, so
`DECISIONS.md §2`'s measured batch-size ladder stays valid rather than needing
re-derivation: batch 12 → 5.62 GiB reserved / 6,096 MiB `nvidia-smi` ≈ 71–75% of
8 GB, with two streams and a ResNet-18 backbone. That ladder was measured, not
estimated, and it is the reason not to invent a new resolution. Going to 224×224
would free VRAM this workload does not need — `DECISIONS.md §8` establishes that
**host RAM, not VRAM, is the binding constraint on this box** — while throwing
away the wrist camera's fine detail near the jaws, which is exactly where a
manipulation policy needs resolution.

**Why 30 fps.** It matches the target hardware camera, so no resampling sits
between sim and real. It matches the existing dataset's fps, so the two are
concatenable. And it divides the physics clock exactly (§3), which the 200 Hz
default did not.

**Consequence:** 124 frames ≈ 4.1 s for this episode — slightly longer than ACT's
default `chunk_size=100`, which is deliberate: an episode shorter than one chunk
is awkward to train on.

## 3. The three clocks

| clock | rate | period | decimation |
|---|---|---|---|
| physics | 240 Hz | 4.1667 ms | — |
| control (IK solve + `ctrl` write) | 60 Hz | 16.67 ms | 4 physics steps |
| recording (frames + state) | 30 Hz | 33.33 ms | 2 control ticks |

The menagerie model ships `timestep=0.005` (200 Hz). **200/30 = 6.67 is not an
integer**, so recording ticks would drift against physics steps. `scene_pickplace.xml`
overrides the timestep to 1/240 s; 240/60/30 gives integer decimations of 4 and 2.
Shrinking the timestep is stability-neutral-to-better, so the menagerie solver
tuning (`implicitfast`, elliptic cone, `impratio=10`) still holds.

Verified: recorded timestamp deltas are `min = max = mean = 0.0333333` s, zero jitter.

### The lockstep contract

At each recording tick, in this order: **(1)** read joint state from `mjData`,
**(2)** render both cameras from that same `mjData`, **(3)** compute the action
from (1), **(4)** append the frame, **(5)** write `ctrl` and step physics.

Observation and images therefore come from one identical `mjData` snapshot, and
the action is computed *from* that snapshot and applied *after* it — the ACT
convention. Nothing is read post-step, so an off-by-one is structurally
impossible rather than merely unlikely.

This is pinned by two deterministic tests, not by inspection:

- `test_action_is_exactly_recomputable_from_the_recorded_observation` — `DiffIK.step`
  re-seeds from the configuration handed to it, so it is a pure function of
  `(q, target)`. Recomputing the action from each frame's own observation must
  reproduce the stored action to `1e-9`. Had it been computed from the previous,
  next, or post-step state, this fails.
- `test_images_are_rendered_at_the_recorded_timestamp` — spies on `render_all` and
  asserts the render instants equal the frame timestamps exactly.

> A statistical alignment test was tried first and **discarded as unsound**: cosine
> similarity between commanded and realised joint steps scores a one-frame shift
> *higher* than no shift (0.995 vs 0.949), because the IK target always leads the
> state and the metric only sees direction. It cannot discriminate the thing it
> claims to. Do not reintroduce it.

## 4. Randomisation distribution

One seed governs everything: `numpy.random.default_rng(seed)`. This episode used
**seed 20260807**, reproducible with `--seed 20260807` alone.

The object is sampled in polar coordinates about the arm base:

- radius `r ∈ [0.17, 0.27] m`, drawn as `sqrt(U(r_min², r_max²))` so samples are
  uniform over the annulus **area** rather than clustering at the inner edge
- azimuth `θ ~ U(−0.60, +0.60) rad`
- z fixed at the cube half-height, 0.015 m; orientation fixed to identity

The drop zone is **not** randomised — it is the fixed `target_zone` body in the
scene XML, read at runtime via `Scene.target_pos()`. That body is the single
source of truth; an earlier version duplicated the position in `config.py` and
the green marker drifted 3 cm from the commanded place pose.

For this episode: object at **(0.1909, −0.0648, 0.015)**, `r = 0.2016`,
`θ = −0.3272 rad`, accepted on rejection attempt **1**.

## 5. The reachability check

The polar annulus is only a *prior*. Nothing assumes it is safe.

**The oracle.** A pose counts as reachable only if `mink` actually converges to
both the pregrasp and the grasp pose from the home configuration, within joint
limits, to `POS_TOL = 0.012 m`. `sample_object_xy` rejection-samples against this
oracle and **raises rather than returning an unreachable pose** — on the first run
it refused all 200 attempts, which is how the geometry below was discovered
instead of being silently shipped.

**What the sweep found.** The SO-101 has five actuated DOF available to the
end-effector pose, so a full 6-DOF target is over-constrained. Measured:
position-only IK converges across the whole workspace, but with orientation fully
constrained it converges only below z ≈ 0.05 m. The original plan — a 0.10 m
pregrasp hop and a 0.14 m transport height — sat squarely in the dead band, which
is why every sample was rejected.

**The fix, and why this one.** `IK_ORI_COST = [0.0, 0.5, 0.5]`: per-axis
orientation cost in the end-effector's *local* frame. The gripper's local x-axis
is its approach direction, so zeroing the first component frees roll *about* the
approach axis — the one rotation a symmetric top-down grasp does not care about —
while pinning the approach direction itself.

| orientation cost | reach | approach axis `dot(−z_world)` | verdict |
|---|---|---|---|
| `[0.5, 0.5, 0.5]` (full) | z ≤ 0.05 only | 0.997 | too constrained |
| `[0.5, 0.5, 0.0]` (free local z) | full workspace | **0.545** at z=0.18 | ~57° off vertical — not a top-down grasp |
| **`[0.0, 0.5, 0.5]`** (free roll about approach) | z ∈ [0.02, 0.065] | **0.997–0.998** | **chosen** |

The manoeuvre was then fitted to the *measured* envelope rather than the envelope
forced to fit the manoeuvre: pregrasp +0.035 m, transport at z = 0.055 m, release
at z = 0.045 m. The arm is short; the motion is deliberately shallow.

**Re-measured every run.** `workspace.survey` grid-sweeps 81 points of the
sampling region plus the drop zone and writes the result to
`reachability_survey.json`. This episode: **81/81 reachable (1.00)**, drop zone
reachable at 0.0109 m error. The claim is a number, not an assumption.

## 6. The weld grasp, and what it cannot tell you

A MuJoCo `<weld>` equality between the `gripper` and `object` bodies, compiled
inactive, toggled at phase boundaries through **`mjData.eq_active`** — runtime
data, not model edits.

On attach, `eq_data[3:10]` is recomputed to the *live* relative pose. Without
that re-anchoring the object teleports to its model-build pose the instant the
constraint activates. Verified: attach displacement **0.000000 m**.

`grasp_mechanism: "weld_constraint"` and its limitation note are written into
`episode_metadata.json`, beside the data rather than only here:

> Grasp success is imposed, not physically simulated. Nothing reads contact
> forces, so **contact precision and grasp robustness are not measurable from
> this data.**

The jaw is nonetheless a real actuated DOF and does collide with the cube — in
the trajectory plot it stalls at ≈0.18 rad against a commanded 0.05, because the
cube blocks it. That closure is physical; the *attachment* is not.

## 7. Phase termination

Every phase ends on a measured Cartesian error (`POS_TOL = 0.012 m`), never on a
step count. The per-phase budget exists only to trip a timeout, and a phase that
exhausts it ends the episode as a **recorded failure** with
`failure_reason: "<phase>_timeout"` — the runner never hangs and never silently
retries. A failed episode is still written to disk; the exit code reports the
truth (`0` success, `2` failure).

| phase | waypoints | gripper | budget (control ticks) |
|---|---|---|---|
| approach | pregrasp → grasp | open | 400 |
| grasp | hold grasp, settle 12 | closed → weld on | 120 |
| transport | lift → above target → release | closed | 500 |
| release | hold, settle 36 | weld off → open | 120 |

`RELEASE_SETTLE_STEPS` is 36 (0.6 s) because at 12 the episode ended while the
jaw was still opening — visible in the first trajectory plot as the commanded
gripper stepping to 1.2 while the measured state only reached 0.85 at the last
frame.

## 8. Reproducing this episode

```bash
MUJOCO_GL=egl uv run python scripts/run_episode.py     --seed 20260807
MUJOCO_GL=egl uv run python scripts/inspect_episode.py --seed 20260807
MUJOCO_GL=egl uv run python -m pytest tests/ -q
```

**`MUJOCO_GL=egl` is required.** On this WSL2 box `egl` and `glfw` both render
correctly and bit-identically; `osmesa` fails at import (`libosmesa6` is not
installed). `egl` is the true headless path and does not depend on WSLg being up.
`scripts/render_probe.py` is the standalone gate: it renders both cameras and
rejects a frame that is black, near-uniform, or has under 50 distinct colours,
because a dead GL backend on WSL2 returns a garbage buffer rather than raising.
