from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


security = HTTPBasic()


def auth_user(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    username = os.getenv("STOCKS_API_USERNAME", "")
    password = os.getenv("STOCKS_API_PASSWORD", "")
    if not username or not password:
        raise HTTPException(status_code=503, detail="dedicated Stocks API authentication is not configured")
    valid = secrets.compare_digest(credentials.username.encode(), username.encode())
    valid = valid and secrets.compare_digest(credentials.password.encode(), password.encode())
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
