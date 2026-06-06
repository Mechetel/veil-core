import base64

from app.celery_app import celery
from app.tasks.callbacks import post_callback

# ML modules are imported lazily inside tasks so the API/worker import stays light
# and torch is only loaded in the worker process that actually runs inference.


@celery.task(bind=True, name="app.tasks.steganography.encode_task")
def encode_task(self, *, client_ref, model_key, message, image_b64):
    from app.ml import steganography as steg

    job_id = self.request.id
    base = {"core_job_id": job_id, "client_ref": client_ref, "kind": "encode"}
    try:
        stego = steg.encode(model_key, base64.b64decode(image_b64), message)
        payload = {
            **base,
            "status": "succeeded",
            "result": {"model_key": model_key, "message": message},
            "output_image_b64": base64.b64encode(stego).decode(),
        }
    except Exception as exc:  # noqa: BLE001
        payload = {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    post_callback("steganography", payload)
    return payload["status"]


@celery.task(bind=True, name="app.tasks.steganography.decode_task")
def decode_task(self, *, client_ref, model_key, image_b64):
    from app.ml import steganography as steg

    job_id = self.request.id
    base = {"core_job_id": job_id, "client_ref": client_ref, "kind": "decode"}
    try:
        message = steg.decode(model_key, base64.b64decode(image_b64))
        payload = {**base, "status": "succeeded", "result": {"message": message}}
    except Exception as exc:  # noqa: BLE001
        payload = {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    post_callback("steganography", payload)
    return payload["status"]
