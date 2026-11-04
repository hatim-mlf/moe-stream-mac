#!/usr/bin/env bash
# Open a new Claude Code session (all permissions) as the project orchestrator, seeded with the kickoff brief.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec claude --dangerously-skip-permissions "$(cat docs/briefs/23-orchestrator-kickoff.md)"
