#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HANDOFF_DIR="$SCRIPT_DIR/handoffs"
mkdir -p "$HANDOFF_DIR"

if [[ "${RESET_SCENE_SERIAL:-0}" == "1" ]]; then
  echo "1" > "$HANDOFF_DIR/NEXT_SCENE_SERIAL.txt"
fi

echo "Running stage 1 (agent skill prepare, sequential serial state)..."
python3 "$SCRIPT_DIR/agent_skill_scene_loop.py" prepare --handoff-dir "$HANDOFF_DIR"

echo
echo "Stage 1 complete."
echo "If code is final, run:"
echo "  python3 \"$SCRIPT_DIR/agent_skill_scene_loop.py\" mark-ready --handoff-dir \"$HANDOFF_DIR\""
echo "Then run stage 2 in Wine CMD:"
echo "  cd /d \"%~dp0\""
echo "  02_run_latest_scene.cmd"
