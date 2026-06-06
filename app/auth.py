import secrets

from fastapi import Header, HTTPException, status

from .config import settings


def require_token(x_auth_token: str | None = Header(default=None)) -> None:
    """FastAPI dependency that enforces token auth on protected routes.

    Compares the ``X-Auth-Token`` header against ``settings.veil_core_token``
    using a constant-time comparison. Raises 401 on mismatch or absence.
    """
    if x_auth_token is None or not secrets.compare_digest(
        x_auth_token, settings.veil_core_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Auth-Token",
        )
