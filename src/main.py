import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# Set OPENSSL_CONF environment variable at the very top
os.environ["OPENSSL_CONF"] = "openssl.cnf"

import asyncio

from helpers.logger import logger
from test_crawler import main

if __name__ == "__main__":
    logger.info("Starting crawler...")
    asyncio.run(main())
    logger.info("Crawler finished.")
