#!/usr/bin/env bash
# Two-way sync of main with GitHub: pull cloud-session commits, push local commits.
# Only committed work is pushed; uncommitted changes are stashed and restored.
# On a rebase conflict it aborts, leaves the repo untouched and exits 1.
set -euo pipefail

readonly ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly BRANCH="main"

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

cd "$ROOT"

if [[ -d .git/rebase-merge || -d .git/rebase-apply || -f .git/MERGE_HEAD ]]; then
  log "SKIP: rebase/merge in progress, resolve it manually"
  exit 1
fi

current="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current" != "$BRANCH" ]]; then
  log "SKIP: on branch '$current', sync only runs on $BRANCH"
  exit 0
fi

git fetch --quiet --prune origin

if ! git pull --quiet --rebase --autostash origin "$BRANCH"; then
  git rebase --abort 2>/dev/null || true
  log "CONFLICT: local and remote $BRANCH diverged, run 'git pull --rebase' and resolve"
  exit 1
fi

ahead="$(git rev-list --count "origin/$BRANCH..HEAD")"
if (( ahead > 0 )); then
  git push --quiet origin "HEAD:$BRANCH"
  log "pushed $ahead commit(s)"
fi

log "in sync at $(git rev-parse --short HEAD)"
