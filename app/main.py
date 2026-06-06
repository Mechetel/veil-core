from fastapi import Depends, FastAPI

from .auth import require_token

app = FastAPI(title="Veil Core")


@app.get("/up")
def health() -> dict[str, str]:
    """Open health endpoint for Kamal / Docker healthchecks."""
    return {"status": "ok"}


@app.get("/", dependencies=[Depends(require_token)])
def root() -> dict[str, str]:
    """Token-protected hello endpoint."""
    return {"message": "hello"}
