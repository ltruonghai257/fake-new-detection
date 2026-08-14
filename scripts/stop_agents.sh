#!/usr/bin/env bash
# stop_agents.sh — graceful shutdown of all A2A factcheck agent services.
#
# Reads .pids/<name>.pid files written by start_agents.sh (D-13), sends
# SIGTERM, waits up to 5s, then SIGKILLs stragglers. Always exits 0.
#
# Safety (WR-04): a pid is only signalled after verifying the process command
# line is actually a factcheck agent — a reused/stale pid can never kill an
# unrelated process.

set -u
cd "$(dirname "$0")/.."

count=0
for pidfile in .pids/*.pid; do
  [ -e "${pidfile}" ] || continue
  name=$(basename "${pidfile}" .pid)
  pid=$(cat "${pidfile}")

  proc_cmd=$(ps -p "${pid}" -o command= 2>/dev/null || true)
  case "${proc_cmd}" in
    *"factcheck_agents.agents.${name}"*)
      ;;
    *)
      echo "⚠ ${name}: pid ${pid} is not a factcheck agent (${proc_cmd:-process gone}) — skipping"
      rm -f "${pidfile}"
      count=$((count + 1))
      continue
      ;;
  esac

  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
      echo "⚠ ${name} (pid ${pid}) killed with SIGKILL"
    else
      echo "✓ ${name} (pid ${pid}) stopped"
    fi
  else
    echo "⚠ ${name} (pid ${pid}) already gone"
  fi
  rm -f "${pidfile}"
  count=$((count + 1))
done

echo "✓ Stopped ${count} agents"
exit 0
