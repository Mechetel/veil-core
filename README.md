# Veil Core

FastAPI + Celery compute service for **Veil** — the steganography brain that wraps
the Attention-SteganoGAN neural networks. The `veil-web` Rails UI submits jobs over
HTTP (ActiveResource); this core runs them asynchronously on Celery workers and
**POSTs the result back** to web. Stateless: web owns all persistence.

## Domains & API

All `/v1/*` routes require the `X-Auth-Token` header to equal `VEIL_CORE_TOKEN`.
Job submits return `202 {id, status:"queued"}`; the result is delivered to web via
a callback (see below). `/up` is open for healthchecks.

**Steganography** (`steg` queue)
| Method | Path | Body |
|--------|------|------|
| GET  | `/v1/steganography/models` | — (10 models) |
| POST | `/v1/steganography/encode` | `{model_key, message, image_b64, client_ref}` |
| POST | `/v1/steganography/decode` | `{model_key, image_b64, client_ref}` |
| GET  | `/v1/steganography/jobs/{id}` | — (debug status) |

**Steganalysis** (`analysis` queue)
| Method | Path | Body |
|--------|------|------|
| GET  | `/v1/steganalysis/models` | — (10 detectors) |
| POST | `/v1/steganalysis/analyze` | `{analyzer_key, image_b64, client_ref}` |
| GET  | `/v1/steganalysis/jobs/{id}` | — (debug status) |

**Callback (core → web):** on completion a worker POSTs to
`${WEB_CALLBACK_URL}/callbacks/{steganography,steganalysis}` with header
`X-Auth-Token: VEIL_CALLBACK_TOKEN` and body
`{core_job_id, client_ref, kind, status, result?, output_image_b64?, error?}`.

## Models & weights

`scripts/vendor_ml.sh` copies the `steganogan/` + `steganalyzers/` packages from the
dissertation repo (required: `.steg` files are pickled SteganoGAN objects).
`scripts/collect_weights.py` copies the 10 steg + 10 analyzer weights into
`weights/` (git-ignored). The registry lives in `app/registry/{steg,analyzers}.yaml`.

In production `weights/` is a persistent volume; push it with
`scripts/sync_weights_to_server.sh deploy@HOST` after the first deploy.

## Run

### 1. Bare (no Docker) — `bin/dev`
Like veil-web's `bin/dev`: starts uvicorn + both Celery workers in one terminal
(via honcho). First run creates `.venv` and installs deps incl. the CPU torch wheel.
Needs a local Redis and reads tokens/wiring from `docker/secret-envs` + `docker/envs`
automatically (no env exports).
```bash
brew services start redis      # or: docker run -p 6379:6379 redis:7
bin/dev                        # → uvicorn :8000  +  steg worker  +  analysis worker
```
Manual equivalent (separate shells): `.venv/bin/uvicorn app.main:app --reload --port 8000`,
`.venv/bin/celery -A app.celery_app worker -Q steg -l info`, `… -Q analysis …`.

### 2. Docker Compose (dev) — redis + api + both workers
```bash
docker compose -f docker/dev.yml up
```

### 3. Kamal (prod)
Set tokens in `docker/secret-envs/veil-core-production.env`, the registry password
in `docker/secret-envs/docker-hub.env`, point the hosts in `config/deploy.yml`, then
`kamal setup`. After the first deploy: `scripts/sync_weights_to_server.sh deploy@HOST`.
GPU build: `kamal build` with `builder.args.TORCH_INDEX: cu121` + `CUDA: "true"`.

## Test
```bash
CUDA=false PYTHONPATH=. pytest -q
```
