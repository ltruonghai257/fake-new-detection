# Architecture Decision: Fusion Model vs Agent Orchestration

Research note backing the choice between training one fusion model (Option A)
vs orchestrating the three signals via an agent/tool layer (Option B) for the
final Fake/Real/NEI decision. See `README.md` for stage-by-stage notebook docs.

## Repo state (as of this note)

| component | model | output | maturity |
|---|---|---|---|
| COOLANT (image-caption cross-modal) | ResNet50+PhoBERT feats, SE-gated fusion + variational ambiguity head | binary logits [B,2] → softmax fake_prob; 64-dim aligned CLIP embeddings | Trained. Test acc 0.8507, macro-F1 0.8474 |
| AI-art detection | Frozen CLIP ViT-L/14 + MLP head, BCEWithLogits | single logit → sigmoid ai_generated_score ∈[0,1] | **Never run.** Zero executed cells, no checkpoint on disk |
| ViFactCheck (claim verification) | Fully fine-tuned PhoBERT-base-v2 + linear head | 3-class logits [B,3] (Real/Fake/NEI) | Trained. Test acc 0.7892, macro-F1 0.7892 |
| Fusion (05a MIL attention, best) | Fuses frozen ViFactCheck CLS[768] + COOLANT aligned-CLIP[64] | 3-class logits [B,3] | Trained. Macro-F1 0.8023 (best of 05a/05b/05d/06a) |

Stage-4 early gated-fusion attempt failed near-chance (macro-F1 ≈0.31) — superseded
by the 05/06 MIL-attention line. 06b (decision-level meta-learner), 06c (comparison),
06d (end-to-end finetune) exist as code but were never executed — no checkpoints.

## SOTA context (condensed)

- Multimodal OOC detection: NewsCLIPpings, Fakeddit, COSMOS benchmarks; trend
  CLIP-similarity → MLLM+retrieval reasoning (SNIFFER, arXiv:2403.03170, 2024).
- AI-image detection: universal VLM-feature detectors generalize best across
  generators (Ojha et al., arXiv:2302.10174, 2023); known weak spots: generator
  gap, compression/resize sensitivity.
- Claim verification: FEVER-style retrieve→verify→SUPPORTED/REFUTED/NEI
  (Thorne et al. 2018) is the standard; NEI as a **trained class**, not a
  post-hoc threshold, is the clean way to get abstention.
- Fusion paradigms: late fusion is most modular but risks calibration mismatch
  across heterogeneous components; evidence pipelines have the cleanest native
  NEI; LLM-agent orchestration has best auditability in principle but abstain
  reliability is itself an open problem (Srinivasan et al., arXiv:2402.15610, 2024).
- **No published work fuses all three signal types (cross-modal + synthetic-image
  + fact-verification) into one verdict.** This repo is in open territory either way.

## Recommendation: Option A — extend the existing trained fusion (05a)

Reasoning against Option B (MCP + agent):

- NEI is already a trained class in 05a/ViFactCheck, matching the evidence-pipeline
  best practice — an agent's prompted abstain is a known-weaker mechanism.
- No calibration step exists yet on any component's output. Feeding raw,
  uncalibrated scores (especially the never-trained AI-art sigmoid) into an LLM
  compounds the miscalibration into a false sense of confidence in the rationale.
- Zero orchestration infra exists in this repo; 05a extension is a proven pattern
  (05d already appends a scalar signal to MIL attention) vs building agent
  plumbing from scratch.
- Attention/gate weights (05a, 06a) already give partial interpretability without
  the extra latency/cost of an LLM call per decision.

Biggest blocker for **either** option: AI-art detection has never been trained.
Fix that first regardless of A/B.

## Concrete NEI decision rule

Keep 3-class softmax argmax as primary decision, add explicit override:

```
if max(softmax_probs) < τ or (top1_prob - top2_prob) < ε:
    decision = NEI
else:
    decision = argmax
```

Start τ=0.5, ε=0.1; fit isotonic regression per class on dev logits first (no
calibration step exists today — this is the actual gap, not the fusion
mechanism), then sweep τ/ε on dev by NEI precision/recall, lock, run once on test.

## First-step plan

1. Run 03.5/03.6 for real; eval `ai_generated_score` on COOLANT's actual news-photo
   domain (not just its own GAN-art benchmark) — README's own caveat flags likely
   domain-transfer failure. If it fails: exclude the signal, ship 05a's 2-signal
   fusion as-is.
2. New `06e_extended_mil_fusion.ipynb` — extend `MILAttentionFusionHead` to
   optionally append `ai_generated_score`, mirroring 05d's fake_prob-append pattern.
3. Calibration cell (isotonic/temperature scaling on dev logits) in 06e or 06c.
4. NEI threshold sweep cell implementing the rule above; report NEI precision/recall
   vs plain argmax; lock values; single test-set run.
5. Skip MCP/agent build under this recommendation.

## Caveats

- AI-art domain mismatch (GAN-art training data vs real news photos) may simply
  not transfer — budget for "excluded" as a valid outcome.
- No published reference for 3-signal fusion — expect iteration, no ground truth
  to check against.
- Revisit Option B only if component count grows past ~5, or an audit/regulatory
  need forces natural-language rationale trails — not justified now.
