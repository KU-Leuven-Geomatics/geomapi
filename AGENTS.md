# Geomapi — Codex agent guidelines

## Role
Codex owns: `examples/`, `tests/`, `docs/`.  
Codex has shared read/write access to `geomapi/tools/` (coordinate with Claude before changing shared tool interfaces).  
Codex does **not** touch `geomapi/nodes/`, `geomapi/utils/`, or `geomapi/ontology/` — those are owned by Claude.

---

## Branch rules
- Always branch from `Geomapi2.0`, prefix branches with `codex/` (e.g. `codex/add-alignmenttools-example`)
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
- Note: pushing to `main` triggers the `docs.yml` CI workflow (deploys to GitHub Pages) and `package.yml` (publishes to PyPI). Version bumps + main pushes are high-impact — treat them as such.

---

## File deletion
Never delete any file without explicit user permission. If a file appears unused or redundant, flag it and ask — do not remove it unilaterally.

---

## Code standards

### Tests (`tests/`)
- Use **pytest**; match the existing file naming: `test_<module>.py`
- One test file per source module; group related assertions into descriptive test functions
- Use real test data from `tests/testfiles/` — do not mock internal geomapi classes
- Every new function or class added to `geomapi/` must have at least one corresponding test
- Check coverage with `--cov=geomapi` and aim to keep coverage stable or improving
- Tests must be deterministic — no random seeds, no network calls, no dependency on absolute file paths outside `tests/testfiles/`

### Examples (`examples/`)
- Jupyter notebooks only; keep cells short and well-narrated with markdown
- Each notebook must run top-to-bottom without errors in a clean environment
- Use the smallest test dataset that illustrates the point
- Import geomapi at the top; do not rely on internal/private symbols (`_`)

### Docs (`docs/`)
- Sphinx + MyST; follow the existing RST/MD structure under `docs/source/`
- Run `sphinx-build -b html docs/source/ docs/_build` and confirm zero errors before committing doc changes
- API docs are auto-generated via `sphinx-apidoc` from docstrings — do not manually duplicate API descriptions; improve the source docstrings instead (ask Claude to update them in `geomapi/`)
- Do not modify `docs/source/geomapi/` auto-generated files directly; they are overwritten on each build

### General
- **Python 3.10–3.12**
- Type hints on any new helper code
- No commented-out code; no debug print statements left in
- Match existing naming conventions: snake_case for functions/variables, PascalCase for node classes

---

## Out of scope
- Do not modify `geomapi/nodes/`, `geomapi/utils/`, or `geomapi/ontology/` — those belong to Claude
- Do not modify `.github/workflows/` without user instruction
- Do not modify `setup.cfg`, `pyproject.toml`, or `requirements.txt` without user instruction
