"""
Debug để xem evidence có được truyền đúng cho PhoBERT không
"""

from factcheck_agents.graph import initial_state, build_graph
from factcheck_agents.models.phobert_checker import build_evidence_text
from factcheck_agents.agents import verify_agent
from factcheck_agents.models import PhoBERTChecker

print("=" * 60)
print("DEBUG EVIDENCE TRANSMISSION")
print("=" * 60)

statement = "Thủ tướng chính phủ tung gói hỗ trợ 60000 tỷ để hỗ trợ PNJ"

# Tạo initial state
state = initial_state(statement, language="vi")

print(f"\nInitial state:")
print(f"  Statement: {state['statement']}")
print(f"  Evidence count: {len(state.get('evidence', []))}")

# Test build_evidence_text
evidence_sample = [
    {
        "source": "vnexpress.net",
        "url": "https://vnexpress.net/test",
        "content": "Nghiên cứu nâng gói tín dụng ưu đãi thủy sản lên 60.000 tỷ đồng"
    }
]

evidence_text = build_evidence_text(evidence_sample)
print(f"\nEvidence text from sample:")
print(f"  '{evidence_text}'")

# Test PhoBERT predict trực tiếp
print("\n" + "=" * 60)
print("TEST PHOBERT PREDICT DIRECTLY")
print("=" * 60)

phobert = PhoBERTChecker()
phobert.load()

print(f"\nTest 1: Statement only (no evidence)")
result1 = phobert.predict(statement, "")
print(f"  Label: {result1.get('label')}, Confidence: {result1.get('confidence')}")
print(f"  Probabilities: {result1.get('probabilities')}")

print(f"\nTest 2: Statement + conflicting evidence")
conflicting_evidence = "Chính phủ KHÔNG có gói hỗ trợ 60000 tỷ cho PNJ"
result2 = phobert.predict(statement, conflicting_evidence)
print(f"  Label: {result2.get('label')}, Confidence: {result2.get('confidence')}")
print(f"  Probabilities: {result2.get('probabilities')}")

print(f"\nTest 3: Statement + supporting evidence")
supporting_evidence = "Chính phủ đã tung gói hỗ trợ 60000 tỷ cho PNJ"
result3 = phobert.predict(statement, supporting_evidence)
print(f"  Label: {result3.get('label')}, Confidence: {result3.get('confidence')}")
print(f"  Probabilities: {result3.get('probabilities')}")

print("\n" + "=" * 60)
print("KẾT LUẬN")
print("=" * 60)
print("\nVấn đề có thể:")
print("  1. PhoBERT chỉ có 2 classes (SUPPORTED, REFUTED), không có NEI")
print("  2. Evidence text có thể không được truyền đúng trong pipeline")
print("  3. Model bias: luôn倾向于 SUPPORTED khi evidence không rõ ràng")
print("  4. Cần check xem evidence từ web search có được format đúng không")