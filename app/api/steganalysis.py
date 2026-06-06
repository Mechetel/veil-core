from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_token
from app.celery_app import celery
from app.registry.analyzers import list_analyzers, resolve
from app.schemas.steganalysis import AnalyzeIn, AnalyzerOut, JobOut

router = APIRouter(
    prefix="/v1/steganalysis",
    tags=["steganalysis"],
    dependencies=[Depends(require_token)],
)


@router.get("/models", response_model=list[AnalyzerOut])
def models() -> list[AnalyzerOut]:
    return [
        AnalyzerOut(
            key=a.key,
            arch=a.arch,
            label=a.label,
            training=a.training,
            available=a.available,
        )
        for a in list_analyzers()
    ]


@router.post("/analyze", response_model=JobOut, status_code=202)
def analyze(body: AnalyzeIn) -> JobOut:
    try:
        resolve(body.analyzer_key)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    task = celery.send_task(
        "app.tasks.steganalysis.analyze_task",
        kwargs={
            "client_ref": body.client_ref,
            "analyzer_key": body.analyzer_key,
            "image_b64": body.image_b64,
        },
        queue="analysis",
    )
    return JobOut(id=task.id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobOut)
def job(job_id: str) -> JobOut:
    res = AsyncResult(job_id, app=celery)
    return JobOut(id=job_id, status=res.state.lower())
