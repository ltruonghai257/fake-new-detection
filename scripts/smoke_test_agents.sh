#!/usr/bin/env bash
# smoke_test_agents.sh — validate all 10 A2A agent endpoints.
#
# For each agent (name, port) from factcheck_agents.config.a2a_ports():
#   GET /.well-known/agent.json, assert valid JSON with a matching "name".
# Prints "✓ N/10 agents responding"; exits non-zero if any agent fails.

set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
pass=0
total=0

while read -r name port; do
  total=$((total + 1))
  if curl -sf --max-time 3 "http://127.0.0.1:${port}/.well-known/agent.json" 2>/dev/null \
    | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); assert d.get('name') == '${name}', (d.get('name'), '${name}'); print('✓ ${name} OK')" 2>/dev/null; then
    pass=$((pass + 1))
  else
    echo "✗ ${name} (port ${port}) not responding or card mismatch"
  fi
done < <("$PYTHON" -c "from factcheck_agents.config import a2a_ports; print('\n'.join(f'{k} {v}' for k, v in a2a_ports().items()))")

echo "✓ ${pass}/${total} agents responding"
[ "${pass}" -eq "${total}" ]
