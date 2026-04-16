#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bootstrap this project as a fresh Git repository and push to GitHub.
#
# Usage:
#   # 1) Create an EMPTY repo on GitHub first (no README, no .gitignore, no LICENSE).
#   # 2) Run from the repo root:
#   bash scripts/bootstrap_git.sh git@github.com:<user>/traffic-sign-recognition.git
#
# Or with HTTPS:
#   bash scripts/bootstrap_git.sh https://github.com/<user>/traffic-sign-recognition.git
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <git-remote-url>" >&2
  exit 2
fi
REMOTE_URL="$1"

# Sanity: we must be at the repo root.
if [[ ! -f pyproject.toml || ! -d src/traffic_signs ]]; then
  echo "ERROR: run this from the repository root (pyproject.toml not found)." >&2
  exit 1
fi

# Keep empty tracked directories alive.
touch data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep checkpoints/.gitkeep

if [[ ! -d .git ]]; then
  git init -b main
  echo "Initialised new repository on branch 'main'."
else
  echo "Existing .git directory found — reusing it."
fi

# Guard against committing data.
git config core.autocrlf input || true

git add .
git commit -m "chore: initial public import of traffic-signs project" \
  -m "Refactored from a single Jupyter notebook into a library, CLI and test suite. See CHANGELOG.md."

# Attach remote.
if git remote | grep -q '^origin$'; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

git branch -M main
echo ""
echo "About to push to: $REMOTE_URL"
echo "If this is wrong, Ctrl-C now. Otherwise press Enter."
read -r
git push -u origin main
echo ""
echo "✓ Pushed. Check https://${REMOTE_URL##*github.com[:/]}"
