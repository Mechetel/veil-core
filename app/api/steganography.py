from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_token
from app.celery_app import celery
from app.registry.steg import list_models, resolve
from app.schemas.steganography import DecodeIn, EncodeIn, JobOut, StegModelOut

router = APIRouter(
    prefix="/v1/steganography",
    tags=["steganography"],
    dependencies=[Depends(require_token)],
)


@router.get("/models", response_model=list[StegModelOut])
def models() -> list[StegModelOut]:
    return [
        StegModelOut(
            key=m.key,
            family=m.family,
            label=m.label,
            dataset=m.dataset,
            data_depth=m.data_depth,
            available=m.available,
        )
        for m in list_models()
    ]


@router.post("/encode", response_model=JobOut, status_code=202)
def encode(body: EncodeIn) -> JobOut:
    _validate(body.model_key)
    task = celery.send_task(
        "app.tasks.steganography.encode_task",
        kwargs={
            "client_ref": body.client_ref,
            "model_key": body.model_key,
            "message": body.message,
            "image_b64": body.image_b64,
        },
        queue="steg",
    )
    return JobOut(id=task.id, status="queued")


@router.post("/decode", response_model=JobOut, status_code=202)
def decode(body: DecodeIn) -> JobOut:
    _validate(body.model_key)
    task = celery.send_task(
        "app.tasks.steganography.decode_task",
        kwargs={
            "client_ref": body.client_ref,
            "model_key": body.model_key,
            "image_b64": body.image_b64,
        },
        queue="steg",
    )
    return JobOut(id=task.id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobOut)
def job(job_id: str) -> JobOut:
    res = AsyncResult(job_id, app=celery)
    return JobOut(id=job_id, status=res.state.lower())


def _validate(model_key: str) -> None:
    try:
        resolve(model_key)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
