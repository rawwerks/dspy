BEFORE ANYTHING ELSE: run 'bd onboard' and follow the instructions

## Fork-Only Files (Never Push to Upstream)
Git hooks in `.githooks/` (pre-push, pre-commit, prepare-commit-msg) protect these files from being pushed to stanfordnlp/dspy:
- `AGENTS.md` (this file)
- `.beads/`

**Setup on new machine:** `git config core.hooksPath .githooks`

Local-only files (in `.git/info/exclude`, never committed anywhere).
