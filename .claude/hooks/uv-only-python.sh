#!/usr/bin/env bash
# PreToolUse hook for Bash: deny bare `python` / `python3` invocations and
# instruct the model to use `uv run python` instead. Lets `uv run python ...`,
# `which python`, `command -v python`, `type python`, and `--version`/`-V`
# probes through.
#
# Mechanism: splits the command on segment separators (`;`, `&`, `|`, newline),
# checks the first token of each segment. If any segment's first token matches
# `python(3)?(.N)?`, returns permissionDecision=deny with an explanatory reason.

set -euo pipefail

input=$(cat)
tool_name=$(printf '%s' "$input" | jq -r '.tool_name // ""')
[ "$tool_name" = "Bash" ] || exit 0

cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')

bad=$(printf '%s' "$cmd" | awk '
BEGIN { RS = "[\n;&|]+"; ORS = "" }
{
  sub(/^[ \t]+/, "")
  n = split($0, a, /[ \t]+/)
  if (n < 1) next
  if (a[1] !~ /^python3?(\.[0-9]+)?$/) next
  # Allow harmless probes
  if (n >= 2 && (a[2] == "--version" || a[2] == "-V")) next
  print a[1]
  exit
}
')

if [ -n "$bad" ]; then
  reason="Bare \`$bad\` is forbidden in this project. Use \`uv run python\` instead:
  uv run python path/to/script.py
  uv run python -m foo.bar
  uv run python -c \"...\"
  uv run --with requests python script.py"
  jq -n --arg reason "$reason" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $reason
    }
  }'
fi

exit 0
