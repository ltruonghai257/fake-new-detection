#!/usr/bin/env bash
# start_agents.sh — start all 10 A2A factcheck agent services (ports 9001-9010).
#
# Behavior (see .planning/phases/03 CONTEXT.md):
#   D-09  sequential startup (one at a time) for easy debugging
#   D-10  per-agent uvicorn logs in logs/agent_<name>.log
#   D-11  port already in use is a hard failure — abort immediately
#   D-12  after starting, block until every agent answers /.well-known/agent.json
#   D-13  per-agent .pid files in .pids/ for stop_agents.sh
#
# On any failure (port conflict, readiness timeout) agents started by THIS run
# are terminated and their pid files removed, so a failed run never leaves
# orphans behind (WR-03).
#
# Ports are read from factcheck_agents.config.a2a_ports() so A2A_PORT_* env
# overrides are honored everywhere.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p logs .pids
START_TS=$(date +%s)

STARTED_PIDFILES=()
ALL_READY=0

cleanup_on_failure() {
  if [ "${ALL_READY}" -ne 1 ]; then
    echo "Cleaning up started agent process(es)..." >&2
    for pidfile in "${STARTED_PIDFILES[@]:-}"; do
      [ -f "${pidfile}" ] || continue
      kill -TERM "$(cat "${pidfile}")" 2>/dev/null || true
      rm -f "${pidfile}"
    done
  fi
}
trap cleanup_on_failure EXIT

started=0
while read -r name port; do
  if lsof -ti :"${port}" >/dev/null 2>&1; then
    echo "ERROR: port ${port} (${name}) already in use — aborting (D-11)." >&2
    exit 1
  fi
  "$PYTHON" -m "factcheck_agents.agents.${name}" > "logs/agent_${name}.log" 2>&1 &
  echo $! > ".pids/${name}.pid"
  STARTED_PIDFILES+=(".pids/${name}.pid")
  echo "started ${name} (pid $!, port ${port})"
  started=$((started + 1))
  sleep 0.5
done < <("$PYTHON" -c "from factcheck_agents.config import a2a_ports; print('\n'.join(f'{k} {v}' for k, v in a2a_ports().items()))")

# D-12: block until every agent answers /.well-known/agent.json (max 30s).
deadline=$((SECONDS + 30))
ready=0
while [ "${SECONDS}" -lt "${deadline}" ]; do
  ready=0
  while read -r name port; do
    if curl -sf --max-time 1 "http://127.0.0.1:${port}/.well-known/agent.json" 2>/dev/null | grep -q '"name"'; then
      ready=$((ready + 1))
    fi
  done < <("$PYTHON" -c "from factcheck_agents.config import a2a_ports; print('\n'.join(f'{k} {v}' for k, v in a2a_ports().items()))")
  if [ "${ready}" -eq "${started}" ]; then
    break
  fi
  sleep 0.5
done

if [ "${ready}" -ne "${started}" ]; then
  echo "ERROR: only ${ready}/${started} agents ready within 30s — check logs/agent_*.log" >&2
  exit 1
fi
ALL_READY=1
echo "✓ All ${started} agents ready ($(( $(date +%s) - START_TS ))s)"
