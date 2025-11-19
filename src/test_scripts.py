import asyncio
from unittest.mock import patch
import pytest

# Import the main functions from the scripts
from test_crawler import main as crawler_main
from agent_test import main as agent_main

@pytest.mark.asyncio
async def test_crawler_main(monkeypatch):
    """
    Tests the main function from test_crawler.py.
    """
    # Mock input to avoid hanging the test
    monkeypatch.setattr('builtins.input', lambda _: 'no_all')
    await crawler_main()

@pytest.mark.asyncio
async def test_agent_main():
    """
    Tests the main function from agent_test.py.
    """
    await agent_main()
