import os

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import logger

PUBLIC_PATHS = frozenset({"/docs", "/openapi.json", "/health"})

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    logger.error("JWT_SECRET_KEY is not configured")
    raise RuntimeError("JWT_SECRET_KEY must be set")


async def security_middleware(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    authorization = request.headers.get("Authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token:
        logger.warning(
            "Unauthorized request to %s: missing Bearer token", request.url.path
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Bearer token is required"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=["HS256"],
            options={"require": ["exp"]},
        )
    except jwt.ExpiredSignatureError:
        logger.warning("Unauthorized request to %s: expired token", request.url.path)
        return JSONResponse(
            status_code=401,
            content={"detail": "Bearer token has expired"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        logger.warning("Unauthorized request to %s: invalid token", request.url.path)
        return JSONResponse(
            status_code=401,
            content={"detail": "Bearer token is invalid"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    request.state.user = payload
    logger.debug("Authenticated request to %s", request.url.path)
    return await call_next(request)
