import base64

from app.celery_app import celery
from app.tasks.callbacks import post_callback


@celery.task(bind=True, name="app.tasks.steganalysis.analyze_task")
def analyze_task(self, *, client_ref, analyzer_key, image_b64):
    from app.ml import steganalysis as ana

    job_id = self.request.id
    base = {"core_job_id": job_id, "client_ref": client_ref, "kind": "analyze"}
    try:
        result = ana.analyze(analyzer_key, base64.b64decode(image_b64))
        payload = {**base, "status": "succeeded", "result": result}
    except Exception as exc:  # noqa: BLE001
        payload = {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    post_callback("steganalysis", payload)
    return payload["status"]
