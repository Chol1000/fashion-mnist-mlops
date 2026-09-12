#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Deploy the current branch to a Hugging Face Space.
#
# A Space identifies itself through YAML frontmatter at the top of its
# README.md — the Docker SDK declaration and the port. The GitHub README has no
# frontmatter, because GitHub renders it as a stray table above the title. So
# rather than keeping two READMEs in step by hand, this script builds the
# Space's README at deploy time: deploy/space-frontmatter.md followed by the
# real README.
#
# The deploy commit is made as a child of the Space's current HEAD, so this
# fast-forwards and never needs --force — the Space keeps its history.
#
#   bash deploy/push-to-space.sh                      # deploy to the API Space
#   bash deploy/push-to-space.sh CholatemGiet/other   # deploy elsewhere
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SPACE_ID="${1:-CholatemGiet/fashion-mnist-api}"
SPACE_URL="https://huggingface.co/spaces/${SPACE_ID}"
REMOTE="space"
TMP_BRANCH="space-deploy"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Anything uncommitted would be silently left out of the deploy, which is the
# kind of surprise that has you debugging a Space that is running code you
# thought you shipped.
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean. Commit or stash first."
    git status --short
    exit 1
fi

SOURCE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Always return the user to the branch they started on, including on failure.
cleanup() {
    git checkout --quiet "$SOURCE_BRANCH" 2>/dev/null || true
    git branch -D "$TMP_BRANCH" --quiet 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Deploying $SOURCE_BRANCH to $SPACE_URL"

git remote get-url "$REMOTE" >/dev/null 2>&1 \
    && git remote set-url "$REMOTE" "$SPACE_URL" \
    || git remote add "$REMOTE" "$SPACE_URL"

echo "==> Fetching the Space's current state"
git fetch --quiet "$REMOTE" main

# Start from the Space's HEAD so the push fast-forwards, then make the tree
# identical to the source branch.
git checkout --quiet -B "$TMP_BRANCH" "$REMOTE/main"
git rm -rq --cached . >/dev/null
git checkout --quiet "$SOURCE_BRANCH" -- .

# Drop anything the Space still has that this branch no longer does — a stale
# Dockerfile or an old frontend left behind would otherwise keep being built.
git clean -fdq -e .venv -e venv -e node_modules -e frontend/dist

echo "==> Building the Space README"
cat deploy/space-frontmatter.md README.md > /tmp/space-readme.md
mv /tmp/space-readme.md README.md

git add -A
if git diff --cached --quiet; then
    echo "==> Nothing to deploy — the Space already matches this branch."
    exit 0
fi

git commit --quiet -m "Deploy $(git rev-parse --short "$SOURCE_BRANCH") from $SOURCE_BRANCH"

echo "==> Pushing (this uploads the model and dataset via LFS — can take a few minutes)"
git push "$REMOTE" "$TMP_BRANCH:main"

echo ""
echo "==> Done. The Space is rebuilding:"
echo "    Build logs : $SPACE_URL"
echo "    Dashboard  : https://$(echo "$SPACE_ID" | tr '/' '-' | tr '[:upper:]' '[:lower:]').hf.space/"
