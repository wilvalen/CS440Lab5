#!/usr/bin/env bash
# PostToolUse hook (Claude Code): runs after every Edit/Write/MultiEdit.
# If the edited file is a .py file in this repo, it (1) byte-compiles it and
# (2) runs the fast Lab 4 test suite. On failure it exits 2, which sends the
# stderr text back to Claude so it fixes the problem before moving on.
set -u
input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

case "$file" in *.py) ;; *) exit 0 ;; esac
case "$file" in "$root"/*) ;; *) exit 0 ;; esac   # ignore files outside the repo
cd "$root" || exit 0

# aima/mdp.py imports numpy, so pick an interpreter that has it.
PY=""
for cand in "${LAB_PYTHON:-}" python3.13 /opt/anaconda3/bin/python3 python3; do
  if [ -n "$cand" ] && command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import numpy' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "check_python hook: no Python with numpy found (set LAB_PYTHON); skipping checks" >&2
  exit 0
fi

if ! out=$("$PY" -m py_compile "$file" 2>&1); then
  printf 'Syntax error in %s:\n%s\n' "$file" "$out" >&2
  exit 2
fi
if ! out=$("$PY" -m unittest test_qlearning 2>&1); then
  printf 'Lab 4 tests failed after editing %s:\n%s\n' "$file" "$(printf '%s' "$out" | tail -30)" >&2
  exit 2
fi
exit 0
