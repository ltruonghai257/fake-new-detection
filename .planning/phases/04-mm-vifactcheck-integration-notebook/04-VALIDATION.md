---
phase: 4
slug: mm-vifactcheck-integration-notebook
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-12
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | grep-on-notebook-JSON (Phase 3 precedent — no pytest for notebook-builder phases) + optional smoke run via `jupyter nbconvert --execute` |
| **Config file** | `pytest.ini` (project root, unused for this phase) |
| **Quick run command** | `rtk grep -c "<acceptance_string>" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` |
| **Full suite command** | Acceptance grep sweep over all required strings from PLAN.md, then `jupyter nbconvert --to script notebooks/pipeline/04_mm_vifactcheck_integration.ipynb --stdout > /dev/null` to confirm cells parse as valid Python |
| **Estimated runtime** | ~5 seconds (grep + nbconvert parse); smoke notebook run ~3–8 minutes once Phase 2 HDF5 + Phase 3 checkpoint exist |

---

## Sampling Rate

- **After every task commit:** Run the task's grep-acceptance command(s) (each `<acceptance_criteria>` bullet is a one-liner).
- **After every plan wave:** Full grep sweep + `nbconvert --to script` parse check.
- **Before `/gsd-verify-work`:** Full suite must be green AND a smoke run (`CONFIG["safety"]["smoke_test"] = True`) must produce `best_model.pth` + `ablation_table.csv` + `mm_vifactcheck_results.json` + `test_confusion_matrix.png` artifacts under timestamped run dirs (gated on Phase 3 completion).
- **Max feedback latency:** ~5 seconds for grep checks; ~8 minutes for the full smoke run.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | NB-01, NB-02, NB-03 | T-04-01 (no abs paths) | No `/Users/` literals; one `CONFIG = {` | grep | `rtk grep -c "/Users/" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` must == 0; `rtk grep -c '^CONFIG = {' notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` must == 1 | ✅ (after Task 1) | ⬜ pending |
| 04-01-02 | 01 | 1 | MMVF-01, NB-02 | T-04-02 (missing-ckpt fail clearly) | Manifest read + 6 patches + `freeze_for_stage2` assert | grep | `rtk grep -n "checkpoint_manifest.json\|freeze_for_stage2\|patch_clip_projection" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ✅ (after Task 2) | ⬜ pending |
| 04-01-03 | 01 | 1 | MMVF-01, MMVF-02 (resolved per D-03) | T-04-03 (article_id join correctness) | Fail fast on len mismatch; per-pair label join | grep | `rtk grep -n "text_aligned_clip\|image_aligned_clip\|stage2_features" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ✅ (after Task 3) | ⬜ pending |
| 04-01-04 | 01 | 1 | MMVF-03 | T-04-04 (correct mode switch) | All 4 modes share constructor | grep | `rtk grep -n "class GatedFusionHead\|h_text_proj\|h_mm_proj" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ✅ (after Task 4) | ⬜ pending |
| 04-01-05 | 01 | 1 | MMVF-04, MMVF-05 | T-04-05 (no test-time leak from in-memory model) | Reload best before test eval | grep | `rtk grep -n "OneCycleLR\|val_macro_f1\|class_weights\|reload_best" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ✅ (after Task 5) | ⬜ pending |
| 04-01-06 | 01 | 1 | MMVF-06, MMVF-07, NB-03 | T-04-06 (MLflow leak across configs) | Per-config MLflow run with try/finally | grep | `rtk grep -n "ablation_table.csv\|mm_vifactcheck_results.json\|test_confusion_matrix.png" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ✅ (after Task 6) | ⬜ pending |
| 04-01-07 | 01 | 1 | All phase reqs | — | Notebook parses; cells are valid Python | parse | `rtk jupyter nbconvert --to script notebooks/pipeline/04_mm_vifactcheck_integration.ipynb --stdout > /tmp/nb4.py && python -c 'import py_compile; py_compile.compile(\"/tmp/nb4.py\")'` | ✅ (post-Task 6) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

No test framework install needed — Phase 4 uses Phase 3's grep-on-notebook precedent. All Phase 4 verifications are derived from PLAN.md `<acceptance_criteria>` blocks.

- [x] No pytest test files required.
- [x] `pytest.ini` already exists at repo root (used by `tests/` for src-level helpers, irrelevant here).
- [x] `nbconvert` is available with Jupyter (already in environment).

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full ablation run quality (real macro-F1 numbers across 4 configs) | MMVF-04, MMVF-06 | Requires Phase 2 HDF5 + Phase 3 trained checkpoint; full run ≈ hours on real data | After Phases 2 + 3 complete: run notebook end-to-end with `SMOKE_TEST=False`; inspect `ablation_table.csv` for plausible macro-F1 (config D should beat config A and B on validation data) |
| Thesis-quality confusion matrix readability | MMVF-05, NB-03 | Visual judgement | Open `training/stage2_results/test_confusion_matrix.png`; labels readable, normalized counts annotated |
| JSON export schema matches thesis chapter outline | MMVF-07 | Schema judgement | Inspect `mm_vifactcheck_results.json`; verify `ablation_summary` contains all 4 configs and `best_config` has full sklearn report |

---

## Validation Sign-Off

- [x] All tasks have grep-`<acceptance_criteria>` or Wave 0 = N/A
- [x] Sampling continuity: every plan task has at least one grep check (no 3 consecutive tasks without automated verify)
- [x] Wave 0 covers all MISSING references (none required)
- [x] No watch-mode flags
- [x] Feedback latency < 10 s for grep, < 10 min for smoke
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-12 (orchestrator)
