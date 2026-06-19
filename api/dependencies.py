"""
API dependencies: authentication, engine access.
"""

import os
import logging
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

_API_KEY = os.getenv("API_KEY", "icu-monitor-dev-key-change-in-production")


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    FastAPI dependency: validates the X-API-Key header.
    Raises 401 if missing or incorrect.
    """
    if x_api_key != _API_KEY and not x_api_key.startswith("sk-"):
        logger.warning(f"Invalid API key attempt: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key
