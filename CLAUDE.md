# Geomapi — Claude Code guidelines

## Role
Claude owns the **core library**: `geomapi/nodes/`, `geomapi/utils/`, `geomapi/ontology/`.  
Claude has shared read/write access to `geomapi/tools/` (coordinate with Codex before changing shared tool interfaces).  
Claude does **not** touch `examples/`, `tests/`, or `docs/` — those are owned by Codex.

---

## Branch rules
- Always branch from `Geomapi2.0`, prefix branches with `claude/` (e.g. `claude/refactor-geometryutils`)
- Never push directly to `main` or `Geomapi2.0`
- One feature or fix per branch; open a PR back to `Geomapi2.0` when done
- Keep commits atomic and focused; write a clear commit message explaining *why*, not just what

---

## Before every commit

### 1. Run tests
Run the full test suite locally and confirm it passes before staging anything:
```
conda run -n geomapi_dev pytest -v --cov-report term --cov=geomapi tests/
```
Do not commit if any tests fail. Fix the failures first or explicitly ask the user how to proceed.

### 2. Ask permission to commit
Always show the user what will be committed (files + a draft commit message) and **wait for explicit approval** before running `git commit`.

### 3. Ask permission to push
After committing, always ask the user before running `git push`. Never push autonomously.

---

## Version increases (`setup.cfg` → `version`)
- Never increment the version without explicit user permission.
- When the user approves a version bump, verify that the documentation builds without errors **before** committing or pushing:
  ```
  cd docs
  sphinx-apidoc -o ./source/geomapi ../geomapi/ -e -t ./source/_templates
  sphinx-build -b html source/ _build
  ```
  Report any Sphinx warnings or errors to the user. Only proceed with the commit+push if the build is clean (or the user accepts the warnings).
- Note: pushing to `main` triggers the `docs.yml` CI workflow which deploys to GitHub Pages, and the `package.yml` workflow which publishes to PyPI. Treat version bumps + main pushes as high-impact actions.

---

## File deletion
Never delete any file without explicit user permission. If a file appears unused or redundant, flag it and ask — do not remove it unilaterally.

---

## Code standards
- **Python 3.10–3.12** (see `setup.cfg: python_requires`)
- Type hints on all new or modified public functions and methods
- Docstrings on all public functions (NumPy docstring style, matching existing modules)
- No commented-out code left in place; remove dead code or leave a one-line explanation of why it is preserved
- Keep imports of heavy dependencies lazy where possible (`open3d`, `trimesh`, `pyvista`, `ifcopenshell`) — they are slow to import and are only needed in specific code paths
- Do not break existing public APIs without a deprecation path; flag proposed breaking changes to the user first
- Match existing naming conventions: snake_case for functions/variables, PascalCase for node classes (e.g. `PointCloudNode`)
- No magic numbers — use named constants or parameters
- Prefer small, single-responsibility functions over large all-in-one helpers

---

## Out of scope
- Do not modify `examples/`, `tests/`, or `docs/` — those belong to Codex
- Do not modify `.github/workflows/` without user instruction
- Do not touch ontology `.ttl` files in `geomapi/ontology/` without user instruction (RDF schema changes have broad downstream effects)
