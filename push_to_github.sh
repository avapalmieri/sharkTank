#!/usr/bin/env bash
#
# Push this project to the "sharkTank" GitHub repo:
#   https://github.com/avapalmieri/sharkTank
#
# Safe to re-run: it only inits git / creates the remote if they don't
# already exist, and it never force-pushes.
#
# Usage:
#   ./push_to_github.sh                # push to main, prompts for GitHub creds if needed
#   COMMIT_MSG="fix: typo" ./push_to_github.sh
#
# If your GitHub account uses SSH keys instead of HTTPS auth, change
# REMOTE_URL below to: git@github.com:avapalmieri/sharkTank.git

set -euo pipefail

REMOTE_URL="https://github.com/avapalmieri/sharkTank"
REMOTE_NAME="origin"
BRANCH="main"
COMMIT_MSG="${COMMIT_MSG:-Update The Tank}"

cd "$(dirname "$0")"

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed. Install it first." >&2
  exit 1
fi

# --- .gitignore: keep secrets and junk out of the repo -------------------
if [ ! -f .gitignore ]; then
  cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
EOF
  echo "Created .gitignore"
fi

# --- init repo if needed --------------------------------------------------
if [ ! -d .git ]; then
  git init
  git branch -M "$BRANCH"
  echo "Initialized git repo on branch '$BRANCH'"
fi

# --- wire up the remote ---------------------------------------------------
if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi
echo "Remote '$REMOTE_NAME' -> $REMOTE_URL"

# --- stage, commit, push ---------------------------------------------------
git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit — working tree matches the last commit."
else
  git commit -m "$COMMIT_MSG"
fi

# Make sure we're not about to blow away remote history we don't have
# locally — pull with rebase first if the remote branch already exists.
if git ls-remote --exit-code --heads "$REMOTE_URL" "$BRANCH" >/dev/null 2>&1; then
  git pull --rebase "$REMOTE_NAME" "$BRANCH" || {
    echo "Rebase pull failed — resolve conflicts, then run: git push $REMOTE_NAME $BRANCH" >&2
    exit 1
  }
fi

git push -u "$REMOTE_NAME" "$BRANCH"

echo "Pushed to $REMOTE_URL ($BRANCH)"
