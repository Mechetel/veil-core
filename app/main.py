from fastapi import FastAPI

from app.api import steganalysis, steganography

app = FastAPI(title="Veil Core")


@app.get("/up")
def health() -> dict[str, str]:
    """Open health endpoint for Kamal / Docker healthchecks."""
    return {"status": "ok"}


app.include_router(steganography.router)
app.include_router(steganalysis.router)
