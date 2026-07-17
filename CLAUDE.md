# inspect-robots-franka: agent guide

Inspect Robots adapters for real Franka FR3 and Panda arms driven by Physical
Intelligence OpenPI DROID policy servers. The framework lives in
[inspect-robots](https://github.com/robocurve/inspect-robots).

## The one big idea

Inspect Robots swaps a policy and an embodiment. This package ships both:

- `openpi` converts pi05-DROID velocity chunks into absolute joint targets.
- `franka` commands the arm through franky and reads two cameras.

Both declare the same 8-D `joint_pos` contract: seven radians plus one normalized
gripper slot, where 0 is closed and 1 is open.

## Layout

- `src/inspect_robots_franka/`: package modules and local module map.
- `tests/`: fully injected hardware-free tests.
- `plans/0001-franka-openpi-design.md`: accepted binding design.

## Working here

- Set `UV_CACHE_DIR=$PWD/.uv-cache` for every uv command in this workspace.
- Install with `uv venv && uv pip install -e ".[dev]"`.
- Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`,
  and `uv run pytest --cov` before handing off.
- Keep strict mypy and 100% statement and branch coverage.
- Keep optional hardware and OpenPI imports lazy so the package imports with
  only Inspect Robots and NumPy.

## Safety invariants

- `FrankaEmbodiment.step()` always clamps to configured limits without relying
  on an approver.
- The policy integrates DROID velocities before emitting actions. The public
  control mode remains absolute `joint_pos`.
- DROID polarity conversion stays in the policy. The embodiment only sees
  open-positive normalized gripper units.
- Construction performs no hardware, network, camera, or stdin work.
- Success reaches scoring only as `termination_reason="success"`.
- The embodiment declares `SELF_PACED` and sleeps inside `step()`.

## CI and releases

- CI installs from `uv.lock`. Run `uv lock` after dependency changes.
- `ci-ok` must need every blocking job.
- The OpenPI client is installed only from the Physical Intelligence git URL.
- Versions come from git tags through hatch-vcs. Do not add a static project
  version to `pyproject.toml`.

## Writing style

- Do not use em dashes in prose. Use periods, commas, colons, or parentheses.
- Use bold only for definition-list leads and critical safety imperatives.
- Do not use decorative emoji, slogans, chiasmus, or "not just X, but Y".
- Headers use colons, never em dashes or italics.
