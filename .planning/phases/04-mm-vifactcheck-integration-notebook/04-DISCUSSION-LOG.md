# Phase 4: MM-ViFactCheck Integration Notebook — Discussion Log

**Date:** 2026-05-12
**Phase:** 4 — MM-ViFactCheck Integration Notebook
**Areas discussed:** Training strategy, Label space & HDF5 input, Ablation table design, Checkpoint selection & export

---

## Area 1: Training Strategy

**Q: The ROADMAP says 'Gated fusion module (h_nli_proj + h_mm_proj)'. The legacy notebook fine-tunes COOLANT's existing detection head directly. Which approach should Phase 4 implement?**
Options presented: Legacy approach (2a frozen + 2b full fine-tune) / Gated fusion as specified / You decide
**Selection:** You decide

*Claude's resolution:* Gated fusion (new GatedFusionHead module on top of frozen COOLANT features). Rationale: cleaner architectural story for thesis committee, better ablation clarity, clean Phase 3→4 handoff with COOLANT fully frozen throughout.

---

**Q: For the gated fusion head — what inputs does it take from the frozen COOLANT checkpoint?**
Options presented: text_aligned_clip + image_aligned_clip only (128+128) / + fake_prob / All Stage 1 outputs
**Selection:** text_aligned_clip + image_aligned_clip only (128-dim each)

---

**Q: Feature extraction strategy — how does Phase 4 get COOLANT features per article?**
Options presented: Extract on-the-fly per batch / Pre-extract all features first, cache to HDF5
**Selection:** You decide

*Claude's resolution:* Pre-extract first. Standard transfer learning pattern — frozen backbone is expensive per batch; cached features enable fast fusion head training and rapid ablation iteration on a GPU instance.

---

**Q: Training hyperparameters for the gated fusion head**
Options presented: 30 epochs / patience 7 (thesis full) / 20 epochs (legacy) / You decide
**Selection:** You decide

*Claude's resolution:* 30 epochs, patience 7, AdamW lr=3e-4, weight_decay=1e-4, label_smoothing=0.1, class-weighted CE, OneCycleLR 5% warmup. Includes SMOKE_TEST=False shortcut.

---

## Area 2: Label Space & HDF5 Input

**Q: Where do ViFactCheck ground-truth labels come from?**
Options presented: Join from JSON at runtime / Re-run Phase 2 with labels in HDF5 / Pre-extract cell creates own enriched HDF5
**Selection:** You decide

*Claude's resolution:* Phase 4 creates its own `stage2_features_{split}.h5` during pre-extraction (join labels from labeled JSONs via article_ids). Doesn't touch Phase 2 outputs. Clean and self-contained.

---

**Q: Label space — binary vs 3-class?**
Options presented: Binary (NEI→Supported) / 3-class / Config-switchable (default binary)
**Selection:** Config-switchable (default binary, NUM_CLASSES in config cell)

---

**Q: Which label variant for the labeled JSON files?**
Options presented: root / nei_as_real / Config-selectable (default root)
**Selection:** You decide

*Claude's resolution:* `root` — consistent with Phase 2 D-12. NEI remapping handled at dataset load time by `NUM_CLASSES`, no separate LABEL_VARIANT config needed in Phase 4.

---

## Area 3: Ablation Table Design

**Q: What should configs A, B, C, D be?**
Options presented: A=text-only / B=image-only / C=concat / D=gated fusion | A=text-only / B=MM-concat / C=gated-frozen / D=gated-finetuned | You decide
**Selection:** You decide

*Claude's resolution:* A=text-only, B=image-only, C=concat (no gating), D=full gated fusion. Classic multimodal ablation. All 4 stay within frozen COOLANT constraint. Tells the strongest thesis story: text alone → image alone → naive combination → learned gating.

---

**Q: Does the notebook train all 4 configs fully or only D?**
Options presented: Train all 4 fully / Train D fully, A/B/C lighter baselines / You decide
**Selection:** You decide

*Claude's resolution:* Train all 4 fully. Fusion head is small and training is fast on cached features. A thesis committee expects real numbers, not handicapped baselines.

---

**Q: How should the ablation table be rendered?**
Options presented: Pandas DataFrame inline / Markdown table / Both (DataFrame + CSV)
**Selection:** You decide

*Claude's resolution:* Both — DataFrame inline AND saved to `training/stage2_results/ablation_table.csv`. Enables `pandas.to_latex()` for direct thesis use.

---

## Area 4: Checkpoint Selection & Export

**Q: Checkpoint selection metric — val macro-F1 or val accuracy?**
Options presented: Val macro-F1 (ROADMAP MMVF-04) / Val accuracy / Both checkpoints
**Selection:** Val macro-F1 (as ROADMAP specifies)

---

**Q: What does the final JSON export contain?**
Options presented: Primary results only (config D) / Full report (all configs + config D detail) / You decide
**Selection:** You decide

*Claude's resolution:* Full report — `ablation_summary` (all 4 configs) + `best_config` (config D full classification report, confusion matrix, hyperparameters, best epoch). Single file for all thesis tables.

---

**Q: Checkpoint save location**
Options presented: training/checkpoints_stage2/{run_name}/ (mirrors Phase 3) / training/stage2/{config_name}/
**Selection:** You decide

*Claude's resolution:* `training/checkpoints_stage2/{config}_{timestamp}/best_model.pth`, results at `training/stage2_results/`. Mirrors Phase 3 conventions exactly.

---

## Deferred Ideas

None — discussion stayed within Phase 4 scope.

---

*Discussion log — Phase 4 MM-ViFactCheck Integration Notebook — 2026-05-12*
