import os

# Set OPENSSL_CONF environment variable at the very top
os.environ["OPENSSL_CONF"] = "openssl.cnf"

import asyncio

from test_crawler import main

if __name__ == "__main__":
    asyncio.run(main())
