# factcheck_agents

A standalone multi-agent module that fact-checks a statement using web evidence
plus this project's two trained models. Design is inspired by
[TradingAgents](https://deepwiki.com/TauricResearch/TradingAgents): specialized
agents share a single state object and run in sequence, orchestrated by LangGraph.

```
User statement (+ optional image)
        │
        ▼
[Search]      LLM drafts queries → Tavily / Google CSE → ranked evidence (truth sources)
        ▼
[Evaluate]    runs the 2 trained models:
              • PhoBERT ViFactCheck  (statement + evidence → SUPPORTED / REFUTED / NEI)
              • COOLANT (optional; only when an image is supplied → REAL / FAKE)
        ▼
[Conclusion]  LLM fuses model verdicts + evidence → TRUE / FALSE / MISLEADING / UNVERIFIED
        ▼
   Verdict + confidence + rationale + citations
```

The module is decoupled from the training pipeline. **Model checkpoints are
loaded lazily and every model failure degrades gracefully** — if a checkpoint
is missing (expected while models are still at the validation stage), that model
reports `unavailable` and the pipeline continues.

## Install

```bash
pip install -r requirements.txt              # base project deps (torch, transformers)
pip install -r factcheck_agents/requirements.txt
```

## Configure (env / `.env`)

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | LLM reasoning for Search + Conclusion agents |
| `FACTCHECK_LLM_MODEL` | optional, defaults to `gpt-4o-mini` |
| `TAVILY_API_KEY` | primary web search |
| `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_ID` | fallback web search |
| `DATA_ROOT` | root holding `training/checkpoints_vifactcheck` and `training/checkpoints_coolant` |
| `VIFACTCHECK_CKPT_DIR` | optional: pin a specific PhoBERT run dir |
| `COOLANT_CKPT_PATH` | optional: pin a specific COOLANT `best_model.pth` |

Without an LLM key the Conclusion agent uses a rule-based fallback; without a
search key the evidence step is empty. Both keep the pipeline runnable.

## Use

CLI:

```bash
python -m factcheck_agents.cli "Vaccine X gây vô sinh"
python -m factcheck_agents.cli "A claim with a photo" --image ./post.jpg --json
```

Python:

```python
from factcheck_agents import run_fact_check
result = run_fact_check("Some claim to verify")
print(result["verdict"])
```

MCP server (stdio):

```bash
python -m factcheck_agents.mcp_server
```

Exposes tools `fact_check`, `search_evidence`, `evaluate_statement`.
```

## Layout

```
factcheck_agents/
  config.py            env-driven settings
  state.py             shared FactCheckState
  models/              PhoBERT + COOLANT wrappers (lazy, graceful)
  tools/web_search.py  Tavily + Google CSE
  agents/              search / evaluate / conclusion nodes
  graph.py             LangGraph wiring
  cli.py               command line
  mcp_server.py        MCP wrapper
```
