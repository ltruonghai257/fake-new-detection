"""
Evidence Retrieval for ViFactCheck Stage 2.

Hybrid SBERT + BM25 retriever. Ported and cleaned from:
  ViFactCheck/src/retrieval/SBERT_BM25.ipynb

Pipeline per sample:
  context text → sentence split → BM25 top-10 → SBERT re-rank → top-5 evidence sentences
"""

import math
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from underthesea import sent_tokenize as vi_sent_tokenize
    _UNDERTHESEA_AVAILABLE = True
except ImportError:
    _UNDERTHESEA_AVAILABLE = False


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Okapi:
    """
    BM25 Okapi ranking. Ported from ViFactCheck retrieval notebook.
    Operates on pre-tokenized (list-of-words) corpus.
    """

    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs: List[dict] = []
        self.idf: dict = {}
        self.doc_len: List[int] = []

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus: List[List[str]]) -> dict:
        nd: dict = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)
            freq: dict = {}
            for word in document:
                freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.append(freq)
            for word in freq:
                nd[word] = nd.get(word, 0) + 1
            self.corpus_size += 1
        self.avgdl = num_doc / self.corpus_size if self.corpus_size else 1
        return nd

    def _calc_idf(self, nd: dict):
        idf_sum = 0.0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf) if self.idf else 0
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: List[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

    def get_top_n(
        self, query: List[str], documents: List[str], n: int = 10
    ) -> Tuple[List[str], List[float]]:
        assert self.corpus_size == len(documents)
        scores = self.get_scores(query)
        min_s, max_s = np.min(scores), np.max(scores)
        scaled = (scores - min_s) / (max_s - min_s + 1e-9)
        top_idx = np.argsort(scaled)[::-1][:n]
        return [documents[i] for i in top_idx], [float(scaled[i]) for i in top_idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    return " ".join(text.strip().lower().split())


def _split_sentences(context: str) -> List[str]:
    if _UNDERTHESEA_AVAILABLE:
        try:
            return [s for s in vi_sent_tokenize(context) if s.strip()]
        except Exception:
            pass
    sentences = re.split(r"[.!?\n]+", context)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# EvidenceRetriever
# ---------------------------------------------------------------------------

class EvidenceRetriever:
    """
    Hybrid SBERT + BM25 evidence retriever for Vietnamese fact-checking.

    Usage:
        retriever = EvidenceRetriever()
        evidence = retriever.retrieve(statement, context, top_n=5)
        # -> List[str] of top-5 evidence sentences

        # Process a whole split:
        enriched = retriever.process_split(records, top_n=5)
        # -> same records with added "evidence_top5" key
    """

    SBERT_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

    def __init__(self, device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModel

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.SBERT_MODEL)
        self.model = AutoModel.from_pretrained(self.SBERT_MODEL).to(device)
        self.model.eval()

    # ------------------------------------------------------------------

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_emb = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def _embed(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        return self._mean_pooling(out, enc["attention_mask"])

    def _sbert_rerank(
        self,
        statement: str,
        candidates: List[str],
        bm25_scores: List[float],
        top_k: int = 5,
    ) -> List[str]:
        from sklearn.preprocessing import MinMaxScaler

        embeddings = self._embed([statement] + candidates)
        claim_emb = embeddings[0].unsqueeze(0)
        cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

        sbert_scores = [
            cosine(claim_emb, embeddings[i + 1].unsqueeze(0)).item()
            for i in range(len(candidates))
        ]

        scaler = MinMaxScaler()
        sbert_norm = scaler.fit_transform(
            np.array(sbert_scores).reshape(-1, 1)
        ).flatten()
        combined = [float(s) * float(b) for s, b in zip(sbert_norm, bm25_scores)]

        top_idx = np.argsort(combined)[::-1][:top_k]
        return [candidates[i] for i in top_idx]

    # ------------------------------------------------------------------

    def retrieve(self, statement: str, context: str, top_n: int = 5) -> List[str]:
        """
        Full pipeline: split → BM25 top-10 → SBERT re-rank → top-n.

        Args:
            statement: The claim text.
            context:   The full article / evidence document.
            top_n:     Number of evidence sentences to return.

        Returns:
            List of up to top_n evidence sentence strings.
        """
        sentences = _split_sentences(context)
        if not sentences:
            return []
        if len(sentences) <= top_n:
            return sentences

        tokenized = [_clean(s).split() for s in sentences]
        query_tokens = _clean(statement).split()

        try:
            bm25 = BM25Okapi(tokenized)
            candidates, bm25_scores = bm25.get_top_n(
                query_tokens, sentences, n=min(10, len(sentences))
            )
        except Exception:
            candidates = sentences[: min(10, len(sentences))]
            bm25_scores = [1.0] * len(candidates)

        return self._sbert_rerank(statement, candidates, bm25_scores, top_k=top_n)

    def process_split(
        self,
        records: List[dict],
        statement_key: str = "Statement",
        context_key: str = "Context",
        top_n: int = 5,
    ) -> List[dict]:
        """
        Add "evidence_top5" to every record in a dataset split.

        Args:
            records:       List of dicts (one per sample).
            statement_key: Key for the claim text.
            context_key:   Key for the evidence document.
            top_n:         Evidence sentences per sample.

        Returns:
            Same records with added "evidence_top5" field.
        """
        results = []
        for rec in tqdm(records, desc="Retrieving evidence"):
            statement = rec.get(statement_key, "")
            context = rec.get(context_key, "")
            evidence = self.retrieve(statement, context, top_n=top_n)
            results.append({**rec, "evidence_top5": evidence})
        return results


# ---------------------------------------------------------------------------
# Convenience: save / load evidence JSON
# ---------------------------------------------------------------------------

def save_evidence(records: List[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records → {path}")


def load_evidence(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
