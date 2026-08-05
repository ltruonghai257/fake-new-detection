"""PhoBERT ViFactCheck text classifier wrapper.

Loads the fine-tuned checkpoint produced by
``notebooks/pipeline/03.9_vifactcheck_training.ipynb``:

    <run>/best_model.pth            (state dict under "model_state_dict")
    <run>/tokenizer/               (saved AutoTokenizer)
    <run>/checkpoint_manifest.json (optional metadata)

The model scores (statement, evidence) -> {Supported, Refuted, NEI}. The
number of classes is inferred from the checkpoint so binary runs also load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..config import settings
from ..state import ModelResult

# label id -> human label for the three_class strategy used in stage 3.9
_LABELS_3 = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}
_LABELS_2 = {0: "SUPPORTED", 1: "REFUTED"}

_TIER_ORDER = {"trusted": 0, "flagged": 1, "social": 2, "unknown": 3}


class PhoBERTClassifier(nn.Module):
    """PhoBERT fine-tuned text classifier for Vietnamese fake-news detection."""

    def __init__(self, backbone_name, num_classes, dropout):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for phobert-base
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [B, 768]
        return self.classifier(cls)  # [B, num_classes]

    def get_cls_features(self, input_ids, attention_mask):
        """Extract frozen [CLS] embeddings (768-dim) for Stage 4 text encoder."""
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # [B, 768]


def _read_manifest_metric(run_dir: Path) -> float:
    """Extract the best validation metric from a run's checkpoint_manifest.json.

    Returns -1.0 if the manifest or metric is missing so that runs without
    manifests sort last but don't crash.

    Handles two manifest formats:
    - Enhanced (03.9_vifactcheck_training.ipynb): ``best_metrics.val_macro_f1``
    - Original (03.9_vifactcheck_original_training.ipynb): top-level ``best_dev_macro_f1``
    """
    man = run_dir / "checkpoint_manifest.json"
    if not man.exists():
        return -1.0
    try:
        data = json.loads(man.read_text())
        # Enhanced notebook: nested best_metrics dict
        metrics = data.get("best_metrics", {})
        for key in (
            "best_val_macro_f1",
            "val_macro_f1",
            "val_accuracy",
            "best_val_accuracy",
        ):
            val = metrics.get(key)
            if val is not None:
                return float(val)
        # Original notebook: top-level metric key
        for key in ("best_dev_macro_f1", "best_val_macro_f1"):
            val = data.get(key)
            if val is not None:
                return float(val)
        return -1.0
    except Exception:
        return -1.0


def _resolve_run_dir() -> Optional[Path]:
    """Explicit override wins; else pick the best run across both checkpoint roots.

    Searches both ``checkpoints_vifactcheck/`` (enhanced, 03.9 notebook) and
    ``checkpoints_vifactcheck_original/`` (original paper, 03.9_original notebook),
    picks the run with the highest validation macro-F1. Falls back to newest-by-mtime
    if no manifests are found.
    """
    # Explicit overrides (enhanced then original)
    for env_path in (settings.phobert_ckpt_dir, settings.phobert_original_ckpt_dir):
        if env_path:
            p = Path(env_path)
            return p if (p / "best_model.pth").exists() else None

    # Auto-discover from both checkpoint roots
    runs: list[Path] = []
    for root in (
        settings.phobert_search_root(),
        settings.phobert_original_search_root(),
    ):
        if root.exists():
            runs += [
                d
                for d in root.iterdir()
                if d.is_dir() and (d / "best_model.pth").exists()
            ]
    if not runs:
        return None

    scored = [(_read_manifest_metric(d), d) for d in runs]
    best_metric = max(s for s, _ in scored)
    if best_metric > 0:
        return max(scored, key=lambda pair: pair[0])[1]

    return max(runs, key=lambda d: d.stat().st_mtime)


def _pick_device() -> str:
    import torch

    if settings.device != "auto":
        return settings.device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_arch(manifest: dict, state: dict) -> str:
    """Return ``'cls'`` or ``'pooler'`` from manifest metadata or state dict keys.

    - ``'cls'``: 03.9_vifactcheck_training.ipynb — ``backbone.`` prefix, ``classifier.1.weight``
    - ``'pooler'``: 03.9_vifactcheck_original_training.ipynb — ``phobert.`` prefix, ``linear.weight``
    """
    arch_str = manifest.get("architecture", "")
    if "pooler" in arch_str.lower():
        return "pooler"
    if any(k.startswith("phobert.") for k in state):
        return "pooler"
    if "linear.weight" in state and not any(k.startswith("backbone.") for k in state):
        return "pooler"
    return "cls"


class PhoBERTChecker:
    """Lazy singleton-style wrapper around the fine-tuned PhoBERT classifier."""

    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self._labels = _LABELS_3
        self._max_length = 256
        self._device = "cpu"
        self._load_error: Optional[str] = None

    # ── loading ──────────────────────────────────────────────────────────
    def _build_cls_model(self, backbone: str, num_classes: int, dropout: float):
        """CLS-token classifier (03.9_vifactcheck_training.ipynb).

        State dict keys: ``backbone.*``, ``classifier.1.weight/bias``.
        """
        return PhoBERTClassifier(backbone, num_classes, dropout)

    def _build_pooler_model(self, backbone: str, num_classes: int, dropout: float):
        """Pooler-output classifier (03.9_vifactcheck_original_training.ipynb).

        Mirrors the original ``PhoBERTClassifier`` from ``plm_training.py``.
        State dict keys: ``phobert.*``, ``linear.weight/bias``.
        """
        import torch.nn as nn
        from transformers import AutoModel

        class _PoolerClassifier(nn.Module):
            def __init__(self, backbone_name, num_classes, dropout):
                super().__init__()
                self.phobert = AutoModel.from_pretrained(backbone_name)
                self.dropout = nn.Dropout(dropout)
                self.linear = nn.Linear(self.phobert.config.hidden_size, num_classes)

            def forward(self, input_ids, attention_mask):
                _, pooled = self.phobert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                return self.linear(self.dropout(pooled))

        return _PoolerClassifier(backbone, num_classes, dropout)

    def load(self) -> bool:
        if self._loaded:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from transformers import AutoTokenizer

            run_dir = _resolve_run_dir()
            if run_dir is None:
                self._load_error = (
                    "No PhoBERT checkpoint found. Set VIFACTCHECK_CKPT_DIR / "
                    "VIFACTCHECK_ORIGINAL_CKPT_DIR or place a run under "
                    f"{settings.phobert_search_root()} or "
                    f"{settings.phobert_original_search_root()}."
                )
                return False

            manifest = {}
            man_path = run_dir / "checkpoint_manifest.json"
            if man_path.exists():
                manifest = json.loads(man_path.read_text())

            ckpt = torch.load(run_dir / "best_model.pth", map_location="cpu")
            state = (
                ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            )

            # backbone: prefer saved value in checkpoint, then manifest, then default
            backbone = (
                (ckpt.get("backbone") if isinstance(ckpt, dict) else None)
                or manifest.get("backbone")
                or manifest.get("model", {}).get("backbone")
                or "vinai/phobert-base-v2"
            )

            # max_length: check all known manifest locations
            self._max_length = int(
                manifest.get("max_length")
                or manifest.get("data", {}).get("max_length")
                or manifest.get("training_setup", {}).get("max_length")
                or 256
            )

            # num_classes: prefer saved value in checkpoint, then infer from head weight
            num_classes = (
                int(ckpt.get("num_classes", 0)) if isinstance(ckpt, dict) else 0
            )
            if num_classes == 0:
                for key in (
                    "classifier.1.weight",
                    "classifier.weight",
                    "linear.weight",
                ):
                    if key in state:
                        num_classes = state[key].shape[0]
                        break
                else:
                    num_classes = 3
            self._labels = _LABELS_2 if num_classes == 2 else _LABELS_3

            dropout = float(
                manifest.get("dropout")
                or manifest.get("model", {}).get("dropout")
                or 0.3
            )

            # Detect architecture and build the matching model class
            arch = _detect_arch(manifest, state)
            self._device = _pick_device()
            if arch == "pooler":
                model = self._build_pooler_model(backbone, num_classes, dropout)
            else:
                model = self._build_cls_model(backbone, num_classes, dropout)
            model.load_state_dict(state, strict=True)
            model.to(self._device).eval()
            self._model = model

            tok_dir = run_dir / "tokenizer"
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(tok_dir) if tok_dir.exists() else backbone
            )
            self._loaded = True
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self._load_error = f"PhoBERT load failed: {exc}"
            return False

    # ── inference ─────────────────────────────────────────────────────────
    def predict(
        self, statement: str, evidence_text: str = "", evidence_count: int = 0
    ) -> ModelResult:
        if not self.load():
            return ModelResult(
                model="phobert_vifactcheck",
                available=False,
                note=self._load_error or "unavailable",
            )
        try:
            import torch
            import torch.nn.functional as F

            enc = self._tokenizer(
                statement,
                evidence_text or None,
                max_length=self._max_length,
                padding="max_length",
                truncation="only_second" if evidence_text else True,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(enc["input_ids"], enc["attention_mask"])
                probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

            label_id = int(max(range(len(probs)), key=lambda i: probs[i]))
            prob_map = {
                self._labels.get(i, str(i)): round(p, 4) for i, p in enumerate(probs)
            }

            # Build workflow steps for UI
            workflow_steps = [
                {
                    "step": "1. Build evidence context",
                    "description": "Concatenate evidence snippets (trusted first) into single passage",
                    "input": f"{evidence_count} evidence items",
                    "output": f"{len(evidence_text or '')} chars evidence text",
                },
                {
                    "step": "2. Tokenize",
                    "description": "Encode statement + evidence with PhoBERT tokenizer (max_length=256)",
                    "input": f"Statement ({len(statement)} chars) + Evidence ({len(evidence_text or '')} chars)",
                    "output": f"input_ids shape: {enc['input_ids'].shape}",
                },
                {
                    "step": "3. Encode",
                    "description": "Pass through PhoBERT backbone to get pooled output",
                    "input": f"input_ids, attention_mask (device: {self._device})",
                    "output": f"logits shape: {logits.shape}",
                },
                {
                    "step": "4. Classify",
                    "description": "Apply softmax to logits to get probabilities",
                    "input": f"logits -> softmax",
                    "output": f"probabilities: {prob_map}",
                },
            ]

            return ModelResult(
                model="phobert_vifactcheck",
                available=True,
                label=self._labels.get(label_id, str(label_id)),
                label_id=label_id,
                probabilities=prob_map,
                confidence=round(probs[label_id], 4),
                note="statement scored against retrieved evidence",
                evidence_text=evidence_text,
                workflow_steps=workflow_steps,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ModelResult(
                model="phobert_vifactcheck",
                available=False,
                note=f"inference error: {exc}",
            )


def build_evidence_text(
    evidence: List[dict], statement: str = "", max_chars: int = 2000
) -> str:
    """Concatenate evidence snippets into a single evidence passage.

    Improvements:
    1. Add metadata: [vnexpress.net] snippet...
    2. Format: each snippet on separate line with source prefix
    3. Truncate at complete sentence (not mid-sentence)
    4. Deduplicate: remove very similar snippets (cosine similarity > 0.9)
    5. Weight by score: prioritize higher score snippets
    6. Group by topic: cluster similar snippets together
    7. Add statement prefix: "Claim: X. Evidence: ..."

    Trusted-tier snippets are placed first so PhoBERT sees the most
    reliable context at the front of the evidence passage.
    """
    import re
    from difflib import SequenceMatcher

    # 1. Sort by tier (trusted first), then by score (higher first)
    evidence = sorted(
        evidence,
        key=lambda e: (
            _TIER_ORDER.get(e.get("source_tier", "unknown"), 3),
            -e.get("score", 0.0),
        ),
    )

    # 2. Deduplicate: remove very similar snippets
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    deduped = []
    for e in evidence:
        snippet = (e.get("snippet") or e.get("content") or "").strip()
        if not snippet:
            continue
        # Check against existing deduped snippets
        is_duplicate = False
        for existing in deduped:
            if _similarity(snippet, existing) > 0.9:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(snippet)

    # 3. Group by topic: simple keyword-based clustering
    def _extract_keywords(text: str) -> set:
        # Extract words, remove common stop words
        words = re.findall(r"\b\w{3,}\b", text.lower())
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "her",
            "was",
            "one",
            "our",
            "out",
            "with",
            "this",
            "that",
            "và",
            "của",
            "đã",
            "có",
            "được",
            "là",
            "cho",
            "trên",
            "với",
            "không",
            "đang",
            "sẽ",
            "các",
            "này",
            "những",
            "người",
        }
        return {w for w in words if w not in stop_words}

    # Simple clustering: group snippets with overlapping keywords
    grouped = []
    used = set()
    for i, snippet in enumerate(deduped):
        if i in used:
            continue
        group = [i]
        keywords_i = _extract_keywords(snippet)
        for j in range(i + 1, len(deduped)):
            if j in used:
                continue
            keywords_j = _extract_keywords(deduped[j])
            if keywords_i & keywords_j:  # Overlapping keywords
                group.append(j)
                used.add(j)
        grouped.append(group)
        used.add(i)

    # Flatten groups, maintaining order
    ordered_snippets = []
    for group in grouped:
        ordered_snippets.extend([deduped[i] for i in group])

    # 4. Build formatted lines with source prefix
    lines = []
    for e in evidence:
        snippet = (e.get("snippet") or e.get("content") or "").strip()
        if not snippet:
            continue
        # Extract domain from URL for prefix
        url = e.get("url", "")
        domain = ""
        if url:
            try:
                from urllib.parse import urlparse

                domain = urlparse(url).netloc
            except Exception:
                domain = "unknown"
        lines.append(f"[{domain}] {snippet}")

    # 5. Truncate at complete sentence (not mid-sentence)
    def _truncate_at_sentence(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        # Find last sentence boundary
        for boundary in [".", "!", "?", ".\n", "!\n", "?\n"]:
            last_pos = truncated.rfind(boundary)
            if last_pos > max_len * 0.8:  # Only if boundary is in last 20%
                return truncated[: last_pos + 1]
        return truncated + "..."

    # Join lines with newlines
    evidence_passage = "\n".join(lines)

    # 6. Truncate to max_chars at sentence boundary
    evidence_passage = _truncate_at_sentence(evidence_passage, max_chars)

    # 7. Add statement prefix
    if statement:
        # Reserve space for statement prefix (about 100 chars)
        available_chars = max_chars - len(f"Claim: {statement}\nEvidence: \n")
        if available_chars > 100:
            evidence_passage = _truncate_at_sentence(evidence_passage, available_chars)
            return f"Claim: {statement}\nEvidence: \n{evidence_passage}"
        else:
            # If statement is too long, just return evidence
            return evidence_passage

    return evidence_passage
