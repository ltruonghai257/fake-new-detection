import asyncio
import os

from test_crawler import main

if __name__ == "__main__":
    # Set OPENSSL_CONF environment variable
    os.environ["OPENSSL_CONF"] = "openssl.cnf"
    asyncio.run(main())
