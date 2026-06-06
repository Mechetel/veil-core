"""Endpoint enqueue contract + task behavior. No Redis/torch needed: send_task is
mocked for the endpoint tests, and tasks run via .apply() with ML + callback stubbed."""
import app.ml.steganalysis as ana_ml
import app.ml.steganography as steg_ml
import app.tasks.steganalysis as ana_task
import app.tasks.steganography as steg_task
from app.celery_app import celery


class _FakeAsync:
    id = "task-123"


def test_encode_endpoint_enqueues(client, auth, monkeypatch):
    calls = {}

    def fake_send(name, kwargs=None, queue=None, **kw):
        calls.update(name=name, kwargs=kwargs, queue=queue)
        return _FakeAsync()

    monkeypatch.setattr(celery, "send_task", fake_send)
    r = client.post(
        "/v1/steganography/encode",
        headers=auth,
        json={"model_key": "dense-div2k", "message": "hi", "image_b64": "AAAA", "client_ref": "gid://veil/Encoding/1"},
    )
    assert r.status_code == 202
    assert r.json() == {"id": "task-123", "status": "queued"}
    assert calls["name"] == "app.tasks.steganography.encode_task"
    assert calls["queue"] == "steg"
    assert calls["kwargs"]["model_key"] == "dense-div2k"


def test_analyze_endpoint_enqueues(client, auth, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        celery,
        "send_task",
        lambda name, kwargs=None, queue=None, **kw: calls.update(name=name, queue=queue) or _FakeAsync(),
    )
    r = client.post(
        "/v1/steganalysis/analyze",
        headers=auth,
        json={"analyzer_key": "xunet-stego", "image_b64": "AAAA"},
    )
    assert r.status_code == 202
    assert calls["name"] == "app.tasks.steganalysis.analyze_task"
    assert calls["queue"] == "analysis"


def test_encode_task_calls_back(monkeypatch):
    captured = {}
    monkeypatch.setattr(steg_ml, "encode", lambda key, data, msg: b"PNGDATA")
    monkeypatch.setattr(steg_task, "post_callback", lambda domain, payload: captured.update(domain=domain, payload=payload))
    steg_task.encode_task.apply(
        kwargs={"client_ref": "gid://veil/Encoding/1", "model_key": "dense-div2k", "message": "hi", "image_b64": "AAAA"}
    )
    assert captured["domain"] == "steganography"
    p = captured["payload"]
    assert p["status"] == "succeeded" and p["kind"] == "encode"
    assert p["client_ref"] == "gid://veil/Encoding/1"
    assert "output_image_b64" in p


def test_analyze_task_calls_back(monkeypatch):
    captured = {}
    monkeypatch.setattr(ana_ml, "analyze", lambda key, data: {"prob_stego": 0.9, "label": "stego"})
    monkeypatch.setattr(ana_task, "post_callback", lambda domain, payload: captured.update(domain=domain, payload=payload))
    ana_task.analyze_task.apply(
        kwargs={"client_ref": "gid://veil/Analysis/2", "analyzer_key": "xunet-stego", "image_b64": "AAAA"}
    )
    assert captured["domain"] == "steganalysis"
    assert captured["payload"]["result"]["label"] == "stego"
