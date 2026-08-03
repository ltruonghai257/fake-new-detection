"""
Test Tavily API response format
"""

import requests
import json
from factcheck_agents.config import settings

query = "Thủ tướng chính phủ tung gói hỗ trợ 60000 tỷ PNJ"

payload = {
    "api_key": settings.tavily_api_key,
    "query": query,
    "max_results": 3,
    "search_depth": "advanced",
    "include_answer": False,
}

print("=" * 60)
print("TEST TAVILY API RESPONSE")
print("=" * 60)
print(f"\nQuery: {query}")

try:
    resp = requests.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"\nStatus: {resp.status_code}")
    print(f"Keys in response: {list(data.keys())}")

    if "results" in data:
        print(f"\nNumber of results: {len(data['results'])}")

        for i, r in enumerate(data['results'][:2], 1):
            print(f"\nResult {i}:")
            print(f"  Title: {r.get('title', 'N/A')}")
            print(f"  URL: {r.get('url', 'N/A')}")
            print(f"  Content: {r.get('content', 'N/A')[:100] if r.get('content') else 'EMPTY'}")
            print(f"  Score: {r.get('score', 'N/A')}")
            print(f"  All keys: {list(r.keys())}")

except Exception as e:
    print(f"Error: {e}")