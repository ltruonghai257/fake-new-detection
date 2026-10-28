"""Check COOLANT label direction on real data.

Usage (>= 2 pairs, each "caption|image_path"):
    uv run python scripts/check_coolant_labels.py \
        "Lũ lớn ở miền Trung|img/flood.jpg" "Bóng đá Việt Nam thắng|img/football.jpg"

Every caption is scored against every image. Diagonal = matched pair and must
be REAL (label 0, high P(REAL)); off-diagonal = mismatched and must be FAKE.
Expected summary: matched->REAL rate and mismatched->FAKE rate both well above 50%.
If both are well BELOW 50%, labels are inverted for your data/preprocessing.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factcheck_agents.models.coolant_checker import CoolantChecker  # noqa: E402

pairs = [a.rsplit("|", 1) for a in sys.argv[1:]]
if len(pairs) < 2:
    sys.exit(__doc__)

c = CoolantChecker()
m_ok = m_n = x_ok = x_n = 0
for i, (cap, _) in enumerate(pairs):
    for j, (_, img) in enumerate(pairs):
        r = c.predict(cap, img)
        if not r.get("available"):
            sys.exit(f"COOLANT unavailable: {r.get('note')}")
        matched = i == j
        want = "REAL" if matched else "FAKE"
        ok = r["label"] == want
        m_n += matched
        x_n += not matched
        m_ok += matched and ok
        x_ok += (not matched) and ok
        print(
            f"cap{i} x img{j} {'MATCH   ' if matched else 'MISMATCH'} -> {r['label']:4} "
            f"P(REAL)={r['probabilities']['REAL']:.2f} sim={r.get('similarity_score')} "
            f"{'ok' if ok else 'WRONG'}"
        )
print(f"\nmatched->REAL: {m_ok}/{m_n}   mismatched->FAKE: {x_ok}/{x_n}")
