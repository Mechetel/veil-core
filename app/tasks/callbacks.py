import logging

import httpx

from app.config import settings

log = logging.getLogger(__name__)


def post_callback(domain: str, payload: dict) -> None:
    """POST a finished job's result to veil-web's domain callback.

    domain: "steganography" | "steganalysis"
    """
    url = f"{settings.web_callback_url.rstrip('/')}/callbacks/{domain}"
    headers = {"X-Auth-Token": settings.veil_callback_token}
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
    except Exception:
        log.exception("callback to %s failed (job=%s)", url, payload.get("core_job_id"))
        raise
