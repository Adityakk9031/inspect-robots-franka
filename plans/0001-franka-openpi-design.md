# 0001: Franka embodiment + openpi policy plugin

Status: draft (critique loop in progress)
Issue: #1

## Goal

Ship the Franka sibling of inspect-robots-so101 and inspect-robots-yam: a plugin
package registering a `franka` embodiment (real FR3/Panda arm via franky) and an
`openpi` policy (websocket client for openpi policy servers, pi05-DROID
convention), declaring one shared 8-D `joint_pos` contract so
`inspect_robots.compat.check_compatibility` passes with zero errors and zero
warnings.

Reference material: the mapped framework contract and template delta table in
the session scratchpad (`franka/framework-contract.md`); inspect-robots v0.20.0;
templates at ../inspect-robots-so101 and ../inspect-robots-yam.

## Stack decision (made after driver/policy landscape research)

- **Driver: franky** (`franky-control` on PyPI). Only actively maintained
  pip-wheel driver current with FR3 firmware (bundles libfranka, default
  0.21.2); async preemptible `robot.move(JointMotion(q), asynchronous=True)`
  matches the 15 Hz VLA streaming pattern (Ruckig re-plans jerk-limited
  trajectories online, so no manual interpolation ramps are needed);
  `franky.Gripper` covers the Franka Hand; `franky-sim` exists for future
  hardware-free integration tests. franky is LGPL-3.0 with "contact us for
  commercial" README wording: it must stay an **optional extra** loaded lazily
  behind a guided-install seam (`_franky.py`, modeled on yam's `_i2rt.py`),
  never a hard dependency.
- **Policy: openpi websocket server client**. pi05-DROID is the strongest
  off-the-shelf Franka policy; protocol is msgpack-numpy over websocket
  (metadata dict on connect; obs dict in, `{"actions": (H, 8)}` out). Transport
  goes through `openpi-client` (pure Python) as an optional `openpi` extra,
  lazily imported in the default seam. Implementer must verify `openpi-client`
  installs from PyPI; if it is git-only, document a git install like yam does
  for i2rt and keep it out of the extra.
- **Out of scope for v1**: Cartesian EEF mode (yam's kinematics.py machinery),
  joint-velocity checkpoints (pi0-FAST-DROID emits velocities; we target the
  pi05 joint-position convention), RealSense reader extra, franky-sim CI
  integration, GR00T backend, bimanual anything.

## The 8-D contract

Single source of truth in `packing.py` + `config.py` helpers used by BOTH
components (compat-by-construction, template invariant).

- `DIM_LABELS = ("joint1", ..., "joint7", "gripper")` (Franka joints are
  1-indexed in all Franka documentation; keep that convention). `NUM_JOINTS=7`,
  `GRIPPER_IDX=7`, `TOTAL_DIM=8`, `STATE_KEY="joint_pos"`.
- Units: slots 0-6 radians (absolute joint targets), slot 7 normalized 0-1
  gripper with **1 = open, 0 = closed** (framework `CANONICAL_STATE_UNITS`
  convention, same as yam).
- `ActionSemantics(control_mode="joint_pos", rotation_repr="none",
  gripper="continuous", frame="base", dim_labels=DIM_LABELS)`. dim_labels are
  mandatory here (so101 omits them; that is a known gap, not a pattern).
- `StateSpec`: exactly one field `StateField(key="joint_pos", shape=(8,),
  unit="rad+normalized")` (conformance: absolute modes need exactly one
  proprioceptive field shaped `(action.dim,)`).
- Cameras: `exterior_cam`, `wrist_cam` (CameraSpec from config
  `cam_height/width`, default 480x640). Names are plugin-local; the policy maps
  them to DROID wire keys internally.
- Default joint limits (FR3, radians; config-overridable, and the README tells
  Panda owners to override):
  `joint_low  = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065,  0.5445, -3.0159, 0.0)`
  `joint_high = ( 2.7437,  1.7837,  2.9007, -0.1518,  2.8065,  4.5169,  3.0159, 1.0)`
  Implementer verifies these against the current FR3 datasheet values in
  franky's docs before coding them in.
- Default home pose: the Franka "ready" pose
  `(0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 1.0)` (gripper open).
- `control_hz = 15.0` (DROID convention), embodiment self-paced
  (`SELF_PACED`, `_pace()` inside `step()`); policy `control_hz=None` (avoids
  the compat `control_rate` warning; the zero/zero property test locks this).

### Gripper polarity (the contract most likely to be wired backwards)

- Embodiment wire convention: normalized, 1 = open. Observation:
  `wire = clamp(width_m / gripper_max_width, 0, 1)` with
  `gripper_max_width=0.08` (config). Command: `width_m = wire *
  gripper_max_width`, sent via a non-blocking gripper move.
- DROID/openpi convention is inverted: `gripper_position` obs uses 0 = open,
  1 = closed; action gripper likewise. The **policy** converts both directions:
  obs out `droid = 1 - wire`, actions in `wire = 1 - droid`. The embodiment
  never sees DROID units. Both conversions get dedicated tests with asymmetric
  values (e.g. 0.25) so a sign error cannot cancel out.

## Package layout

```
inspect-robots-franka/
├── src/inspect_robots_franka/
│   ├── __init__.py        # exports, __version__ via importlib.metadata
│   ├── CLAUDE.md          # module map (mirrors siblings)
│   ├── packing.py         # pure: constants, DIM_LABELS, validate_dim
│   ├── config.py          # FrankaConfig, OpenpiConfig, shared space builders
│   ├── embodiment.py      # FrankaEmbodiment + Driver protocol + camera reader
│   ├── policy.py          # OpenpiPolicy + infer_fn seam
│   ├── operator.py        # OperatorIO (copy yam's EOF-hardened version)
│   ├── preflight.py       # inspect-robots-franka-preflight CLI
│   ├── _franky.py         # lazy franky loader + FRANKY_INSTALL_COMMAND
│   └── py.typed
├── tests/                 # see test plan
├── plans/0001-franka-openpi-design.md
├── .github/workflows/{ci,canary,release}.yml
├── .pre-commit-config.yaml
├── pyproject.toml / uv.lock / README.md / CLAUDE.md / LICENSE / CITATION.cff
└── .gitignore / .env.example
```

## Module contracts

### packing.py (pure, no optional deps)

Constants above, plus `validate_dim(vec) -> np.ndarray` (yam-strict: require
`ndim == 1` and length 8, no silent flatten), `arm_joints(vec) -> (7,)` and
`gripper(vec) -> float` accessors. No wire-format converters needed (franky
speaks plain arrays; the DROID dict formatting lives in policy.py next to its
wire keys).

### config.py

- `_FromKwargs` mixin with `from_kwargs(**flat)` rejecting unknown keys, and
  yam's `_FLOAT_TUPLE_FIELDS` comma-separated tuple parsing for pose/limit
  fields.
- `FrankaConfig` (frozen): `hostname` (FCI address, required for hardware, no
  default probing), `control_hz=15.0`, `joint_low/joint_high` (defaults above),
  `home_pose` (default ready pose; reset always homes), `rest_pose=None`
  (optional park target on close; None = stay put, Franka holds impedance),
  `relative_dynamics_factor=0.15` (franky speed/accel/jerk scale, conservative
  default), `gripper_max_width=0.08`, `gripper_speed=0.05` (m/s),
  `joints_are_delta=False` (delta checkpoints converted to absolute inside
  `step()`, yam pattern), `unattended=False`, `exterior_cam_device` /
  `wrist_cam_device` (V4L2 paths or numeric indices for the builtin OpenCV
  reader; both or neither, else `ConfigError` at reset), `cam_height=480`,
  `cam_width=640`, `docs_extra=""`.
  `__post_init__` validates: control_hz > 0, limits ordered and length 8,
  home/rest pose inside limits, gripper_max_width > 0, dynamics factor in
  (0, 1].
- `OpenpiConfig` (frozen): `host="127.0.0.1"`, `port=8000`, `api_key=None`,
  `timeout_s=20.0`, `action_horizon=16` (pi05-DROID chunk length; recorded in
  `PolicyConfig`), `replan_interval=8` (execute half the chunk open-loop, DROID
  practice; implementer verifies how `PolicyConfig.replan_interval` is consumed
  by the framework rollout and drops it if unsupported), `name="openpi"` (eval
  log label), `resize_px=224` (server-side expected image size; used by the
  default transport's resize-with-pad).
- Shared builders: `ACTION_SEMANTICS` constant, `action_box(cfg)`,
  `observation_space(cfg)`; both components build their `info` from these.

### embodiment.py

- `Driver` Protocol (runtime_checkable), injected via
  `driver_factory: Callable[[FrankaConfig], Driver]`:
  `read_joints() -> np.ndarray (7,)`, `read_gripper_width() -> float`,
  `move_joints(target: np.ndarray) -> None` (asynchronous, preempting),
  `move_joints_sync(target) -> None` (blocking, for homing/parking),
  `move_gripper(width: float) -> None` (non-blocking), `disconnect() -> None`.
- `_default_driver_factory` (`# pragma: no cover`): builds
  `franky.Robot(hostname)` + `franky.Gripper(hostname)` through `_franky.py`'s
  guided loader; applies `relative_dynamics_factor`; async moves via
  `JointMotion + asynchronous=True`; gripper via `move_async`.
- `FrankaEmbodiment`: inert `__init__(config=None, *, driver_factory=None,
  camera_reader=None, operator=None, poll_end=None, clock=None, sleep_fn=None,
  **flat)`; lazy connect at first `reset()`.
  - `reset()`: connect, home via `move_joints_sync(home)` + gripper open
    (franky Ruckig-plans the motion, no manual `_ramp_to`), stand-clear prompt
    then `operator.wait_ready()` (skipped when `unattended`), return first
    observation.
  - `step()`: `validate_dim` -> optional delta->abs -> **hard clamp to
    joint_low/high (backstop independent of any Approver)** -> `move_joints`
    (7) + `move_gripper` (denormalized) -> `_pace()` -> observe ->
    `poll_end()` / `operator.confirm_success()` -> `StepResult`
    (`termination_reason="success"` is the only success channel).
  - `close()`: idempotent; optional `rest_pose` park via sync move; disconnect
    always attempted, handle cleared even on error.
  - Camera seam: yam-style injected `camera_reader() ->
    {"exterior_cam": HxWx3 uint8, "wrist_cam": ...}` + builtin OpenCV reader
    from the two `*_cam_device` config values (lazy cv2 import). Neither
    configured -> `ConfigError` at `reset()` before any driver connect.
  - `RUNTIME_REQUIREMENTS = ("franky", "cv2")`, `DEVICE_SLOTS` for the setup
    wizard (hostname + two camera slots), `bind_task()` storing the bound
    horizon for the operator status line, `EmbodimentInfo.docs` markdown
    (`_DOCS` + `docs_extra`) naming all 8 dim labels.

### policy.py

- `OpenpiPolicy(config=None, *, infer_fn=None, **flat)`; entry point `openpi`.
- `act(observation)`:
  1. Validate presence of both cameras and `joint_pos` state (helpful errors).
  2. Build the DROID obs dict: `"observation/exterior_image_1_left"`,
     `"observation/wrist_image_left"` (native frames, uint8),
     `"observation/joint_position"` (7,), `"observation/gripper_position"`
     (1,) as `1 - wire`, `"prompt"` = instruction.
  3. `infer_fn(obs_dict) -> {"actions": (H, 8) float}`; validate shape,
     non-empty, H rows -> flip gripper column (`wire = 1 - droid`) -> truncate
     to `action_horizon` -> `ActionChunk(control_hz=15.0 from shared config,
     inference_latency_s measured via injected clock)`.
- `_default_infer` (`# pragma: no cover` transport): lazily imports
  `openpi_client`; `WebsocketClientPolicy(host, port, api_key)`; resize-with-pad
  both images to `resize_px` via `openpi_client.image_tools` before sending
  (custom `infer_fn` owns its own resizing; document this). Guided-install
  error message when `openpi_client` is missing.
- `openpi-seam` CI job (so101's `lerobot-seam` analog): installs
  `.[dev,openpi]` and imports every real symbol `_default_infer` touches, so
  upstream drift is caught without a GPU or server.
- `info.control_hz = None`; `num_inferences` counter; `reset()` stashes the
  instruction.

### operator.py / preflight.py / _franky.py

- operator.py: copy yam's (EOFError/OSError -> `EmbodimentFault`,
  `_drain_stdin`), rename.
- preflight.py: standard `build()` / `run_preflight()` / `main()` with
  `--task/--json/--dry-run`; console script `inspect-robots-franka-preflight`.
- `_franky.py`: `_load_franky()` raising a guided-install error
  (`FRANKY_INSTALL_COMMAND = "pip install franky-control"` plus a pointer to
  franky's firmware-to-wheel table: wheels are pinned per libfranka version and
  the robot's firmware dictates which wheel to install).

## pyproject

- Base deps: `inspect-robots>=0.12`, `numpy>=1.24`,
  `opencv-python-headless>=4.8` (lazily imported; import-hygiene still enforces
  the package imports without it).
- Extras: `franka = ["franky-control>=1.1"]`, `openpi = ["openpi-client"]`
  (subject to the PyPI-availability check above), `dev = [pytest, pytest-cov,
  ruff, mypy, pre-commit, numpy<2.5]`.
- Entry points: `[project.entry-points."inspect_robots.embodiments"] franka =
  "inspect_robots_franka.embodiment:FrankaEmbodiment"`;
  `[project.entry-points."inspect_robots.policies"] openpi =
  "inspect_robots_franka.policy:OpenpiPolicy"`.
- Everything else copied from the siblings: hatchling + hatch-vcs (dynamic
  version, no static version ever) + hatch-fancy-pypi-readme (alert
  substitutions), ruff (line 100, py310, E/F/W/I/UP/B/C4/SIM/RUF/D1, D105/D107
  ignored, tests exempt from D1), mypy strict py3.10 with overrides for
  `franky.*`, `cv2.*`, `openpi_client.*`, coverage `fail_under=100`,
  `branch=true`, standard `exclude_also`.

## CI (yam's hardened skeleton: permissions, timeouts)

Jobs: `quality` (ruff check, format --check, mypy; py3.11), `test`
(ubuntu+macos x py3.11/3.12, `uv sync --locked`, pytest --cov at 100%),
`import-hygiene` (no extras; assert `cv2`, `franky`, `openpi_client`,
`websockets`, `torch` absent; import the package), `openpi-seam` (unlocked
`uv pip install -e ".[dev,openpi]"`, import the transport's real symbols),
`ci-ok` aggregate (`if: always()`, needs all four, jq all-success; the single
required check), `alert-red-main`. `canary.yml` and `release.yml` byte-copied
from yam. Branch ruleset (already active): PR-only, `ci-ok` required, strict
up-to-date, no bypass.

## Test plan (all injected; no hardware, no server, no stdin, no cv2)

- `test_packing.py`: constants, label count/uniqueness, validate_dim strictness
  (ndim!=1, wrong length, list input), accessors.
- `test_config.py`: from_kwargs unknown-key rejection, tuple parsing,
  post-init validation cases, default limits ordered, home inside limits.
- `test_embodiment.py`: inert init; lazy connect; homing call order; clamp
  backstop (command outside limits never reaches driver); delta->abs; pacing
  (injected clock/sleep); gripper denormalization asymmetric-value test;
  camera reader injection + ConfigError when unset; operator success ->
  termination_reason="success"; unattended path; close idempotency +
  disconnect-on-error; bind_task; docs content.
- `test_policy.py`: obs dict wire keys byte-exact; gripper polarity both
  directions with asymmetric values; shape/emptiness validation; truncation to
  action_horizon; instruction threading; num_inferences; helpful errors on
  missing cameras/state.
- `test_operator.py`, `test_preflight.py`, `test_franky.py` (loader error
  message contains install command).
- `test_compat.py`: the zero-errors-zero-warnings property; builtin
  `cubepick-reach` realizable; wrong-dim negative; policy-advertises-control_hz
  negative; delta pairing mismatch -> control_mode error.
- `test_embodiment_docs.py`: every DIM_LABEL appears exactly once in docs
  bullets; docs_extra append semantics; no numeric joint-limit leaks.
- `test_api_snapshot.py`: `__all__` snapshot, entry points resolve via
  `registry.resolve`, `.info.name` matches, `__version__` regex.
- `test_eval_end_to_end.py`: full `eval()` on `cubepick-reach` with fake driver
  + fake infer_fn, asserting success propagation and log metrics.

## README (yam structure, so101 length; house writing style)

Sections: badges/intro (two swappable inputs, sibling links), Install (robot
machine: package + `[franka]` extra + firmware wheel note + RT-kernel
requirement; GPU machine: openpi `serve_policy.py` for pi05-DROID), Preflight,
Run on hardware (config.ini example with camera device paths), Safety (clamp
backstop, gripper polarity conversion table, first-run verification with
e-stop, franky RT requirements, delta-vs-absolute warning), Configuration
(field tables for both configs, joint-space unit table), Development,
Citation, License. CITATION.cff + .env.example included. No em dashes in
prose, no decorative emoji, headers use colons.

## Sequencing

1. Plan critique loop (fresh-context subagent) until no substantive findings.
2. Codex implements from this plan; Fable reviews the diff.
3. Push, PR `Closes #1`, CI green.
4. Fresh-eyes review loop on the PR diff until clean; merge.
5. Post-merge: PyPI trusted-publishing pending publisher (owner action),
   first release cut, CLAUDE.md refresh if drift emerged.
