import asyncio
import ssl
from typing import Optional, Dict, Any

import httpx
from helpers.logger import logger


class BaseClient:
    """
    A wrapper around httpx.AsyncClient to provide robust functions for all methods.
    """

    def __init__(
        self,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        verify: bool = True,
    ):
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.verify = verify

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
            follow_redirects=True,  # Automatically follow redirects
            verify=self.verify,
        )

    async def get(
        self, url: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> httpx.Response:
        """
        Sends a GET request.
        """
        return await self._request("GET", url, params=params, **kwargs)

    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        **kwargs,
    ) -> httpx.Response:
        """
        Sends a POST request.
        """
        return await self._request("POST", url, data=data, json=json, **kwargs)

    async def put(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        **kwargs,
    ) -> httpx.Response:
        """
        Sends a PUT request.
        """
        return await self._request("PUT", url, data=data, json=json, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """
        Sends a DELETE request.
        """
        return await self._request("DELETE", url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        A generic request method with error handling and retry logic.
        """
        retries = 3
        backoff_factor = 0.5
        for i in range(retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and i < retries - 1:
                    wait_time = backoff_factor * (2 ** i)
                    logger.warning(f"HTTP error occurred: {e}. Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                # Do not raise for redirects, as they are handled by the client
                if e.response.status_code not in (301, 302, 307, 308):
                    logger.error(f"HTTP error occurred: {e}")
                    raise
                return e.response  # Return the response for redirects
            except httpx.RequestError as e:
                logger.error(f"An error occurred while requesting {e.request.url!r}: {e}")
                raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()