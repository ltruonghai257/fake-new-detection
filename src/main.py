import os

# Set OPENSSL_CONF environment variable at the very top
os.environ["OPENSSL_CONF"] = "openssl.cnf"

import asyncio

from helpers.logger import logger
from test_crawler import main

if __name__ == "__main__":
    logger.info("Starting crawler...")
    asyncio.run(main())
    logger.info("Crawler finished.")
