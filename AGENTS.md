# Repository Guidelines

## Project Structure & Module Organization

InstinctMJ provides humanoid reinforcement-learning environments built on `mjlab` and integrated with `instinct_rl`. Source lives in `src/instinct_mj/`:

- `tasks/`: locomotion, shadowing, perceptive, and parkour tasks, with environment and RL configurations colocated by task family.
- `envs/`, `managers/`, `sensors/`, `terrains/`, and `motion_reference/`: shared simulation components.
- `scripts/`: training, playback, and motion-data utilities.
- `assets/resources/`: robot XML/MJCF, URDF, and meshes.

Register tasks using `register_instinct_task()` from `tasks/registry.py`. Add `__init__.py` at each package level. Experiment logs belong under `logs/instinct_rl/`.

## Agibot X2 Development Conventions

Follow G1's structure and configuration patterns for all X2 development,
adapting geometry and hardware parameters to X2.

- Define the robot in `src/instinct_mj/assets/agibot_x2.py`; keep XML and meshes
  under `assets/resources/agibot_x2/`. Put task sensors, including cameras, in
  `tasks/parkour/config/agibot_x2/`.
- Use the fixed-head, `torso_link`-rooted MJCF for simulation and motion-reference
  kinematics. Retain the original pelvis-root XML as source material.
- Map joint data by name in native MuJoCo order. X2's reversed waist order is
  roll–pitch–yaw; negate all three waist coordinates when converting from the
  pelvis-root model. Derive X2 actuator and symmetry settings rather than copying G1's.
- Keep motion base poses consistent with the simulation's torso frame. Root
  conversions must preserve world-space body poses, geometry, inertias and camera mounting.
- Follow G1's task factories, observation groups, rewards, terminations,
  registration and train/play interfaces. Document necessary X2-specific differences.
- After structural changes, verify imports, model compilation, joint/root
  conventions and forward-kinematic equivalence.

## Build, Test, and Development Commands

Use Python 3.10–3.13 with the `mjlab` Git revision pinned in `pyproject.toml`.

- `uv sync`: install the package and locked dependencies, including `mjlab` from GitHub.
- `uv run instinct-list-envs`: verify imports and task registration; append `shadowing` to filter results.
- `uv run instinct-train Instinct-Locomotion-Flat-G1-v0`: start locomotion training.
- `uv run instinct-play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>`: play a saved run.
- `pre-commit run --all-files`: run formatting and hygiene checks; install `pre-commit` separately if needed.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and modules, and `PascalCase` classes. Follow Black and isort settings in `pyproject.toml`: 120-character lines and Python 3.10 compatibility. Hooks also run pyupgrade, codespell, and whitespace checks; `.flake8` specifies Google-style docstrings. Preserve public task IDs and CLI entry points unless intentionally changing their behavior.

## Testing Guidelines

Pytest targets `tests/`, but this checkout has no test suite or coverage threshold; `tests/` is Git-ignored. Use `test_*.py` names for new tests and run `uv run python -m pytest` with pytest installed. Coordinate tracking changes when adding a committed suite.

For current changes, compile affected modules with `uv run python -m py_compile <file.py>`, check task registration, and exercise relevant train/play workflows. Report commands, outcomes, and outstanding manual validation.

## Commit & Pull Request Guidelines

Use Conventional Commits, consistent with recent history: `feat: add ...`, `refactor: simplify ...`, or `docs: document ...`; optional scopes are allowed. Use imperative descriptions.

Keep PRs focused, explain behavior changes, link related issues, update affected documentation, and include validation results. Discuss large changes in an issue first. Read `CONTRIBUTING.md` and agree to `CONTRIBUTOR_AGREEMENT.md` before contributing.
