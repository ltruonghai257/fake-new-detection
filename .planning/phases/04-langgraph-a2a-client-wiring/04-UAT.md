---
status: complete
phase: 04-langgraph-a2a-client-wiring
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md]
started: 2026-08-17T00:00:00Z
updated: 2026-08-18T13:44:18Z
---

## Current Test

<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. Full Pipeline — All 10 Agents Running

expected: Start all agents with `scripts/start_agents.sh` — it prints "All 10 agents ready". Then run `python -m factcheck_agents.cli "The Great Wall of China is visible from space with the naked eye" --language en`. The command exits 0 within ~2 minutes and prints a verdict (FAKE / REAL / UNVERIFIED) with a rationale and confidence score. The rationale contains real reasoning and citations — no "agent unavailable" degrade notices.
result: pass

### 2. Graceful Degrade — All Agents Down

expected: With all agents stopped (`scripts/stop_agents.sh`), run the same CLI command. It exits 0 (no crash/traceback), prints verdict UNVERIFIED with confidence 0.0, and includes a notice that agents are unavailable.
result: pass

### 3. Partial Outage — Judge Down

expected: Stop only the judge agent (`kill $(cat .pids/judge.pid)`). Run the CLI command. It still exits 0 — the judge step degrades with a notice instead of crashing the pipeline.
result: pass

### 4. Partial Debate — One Advocate Down

expected: Stop only the real advocate (`kill $(cat .pids/real_advocate.pid)`). Run the CLI with a claim likely to trigger debate. The pipeline completes with exit 0 — debate runs one-sided with the fake advocate only, no crash.
result: pass

### 5. JSON Output

expected: `python -m factcheck_agents.cli "<claim>" --json` prints a single JSON object with verdict and confidence fields (parseable with `python -m json.tool`), exit 0.
result: pass

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
