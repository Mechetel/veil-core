# veil-core — FastAPI compute service for Veil

Stateless steganography "brain": wraps the Attention-SteganoGAN networks (PhD
dissertation, `~/Projects/phd_dissertation/state_3/Attention-Steganogan`).
The Rails UI (`~/Projects/veil-web`) submits jobs over HTTP; Celery workers run
them and **POST results back** to web's `/callbacks/*`. Web owns all persistence.

## Run / test

```bash
bin/dev                                  # bare: uvicorn :8000 + both celery workers (honcho)
                                         # needs local Redis (brew services start redis)
docker compose -f docker/dev.yml up      # same stack fully in Docker (redis included)
.venv/bin/pytest tests/                  # unit tests (api, enqueue, registry)
```

Deployment (Kamal, weights volume, IP-only setup): see **DEPLOY.md**.

## Architecture

```
app/
├── main.py            # FastAPI app; open GET /up healthcheck
├── auth.py            # X-Auth-Token == VEIL_CORE_TOKEN guard on all /v1/*
├── config.py          # pydantic-settings; env > dev env_files > defaults
├── celery_app.py      # broker/result = REDIS_URL; routes: steganography→steg, steganalysis→analysis
├── api/               # routers: /v1/steganography/*, /v1/steganalysis/* (202 {id, queued})
├── tasks/             # celery tasks + callbacks.py (POSTs results to WEB_CALLBACK_URL
│                      #   with X-Auth-Token: VEIL_CALLBACK_TOKEN)
├── registry/          # steg.yaml / analyzers.yaml — model key → weight file mapping
├── ml/                # checkpoint.py (CPU-safe .steg unpickler), lru_cache model loading
└── schemas/           # pydantic request/response models
steganogan/, steganalyzers/   # VENDORED from the dissertation repo — see below
weights/               # NOT in git, NOT in the image (volume in prod)
```

## Invariants & gotchas

- **Python is pinned to 3.12** (Dockerfiles, bin/dev): numpy 1.26.4 and
  scikit-image 0.22.0 ship no 3.13 wheels. Don't bump Python or these pins
  independently. torch/torchvision are deliberately **unpinned (latest)**.
- Every `torch.load` must pass `weights_only=False` (.steg files are pickled
  whole models; torch ≥2.6 defaults to True and would break loading).
- `steganogan/` and `steganalyzers/` are **vendored** via
  `scripts/vendor_ml.sh` (rsync from the dissertation repo). Don't hand-edit
  except the documented local patch (decoder_service basename error messages) —
  re-vendoring overwrites local changes.
- Checkpoints unpickle under the original `steganogan.*` module path; the
  packages must stay importable from the repo root (`PYTHONPATH=/app` or `.`).
- Loaded models are cached with `lru_cache` — after replacing weight files the
  process must restart to pick them up (prod: `kamal app boot`).
- Adding a model = registry YAML entry (key, weight filename) + weight file in
  `weights/steg/` or `weights/analyzers/`; models lazy-load on first use.
- Tokens: `VEIL_CORE_TOKEN` (inbound, from web) and `VEIL_CALLBACK_TOKEN`
  (outbound, to web) must match veil-web's `docker/secret-envs/` copies.
- `docker/secret-envs/*production.env` + `docker-hub.env` are **git-crypt
  encrypted** and tracked in git on purpose — never gitignore them, never add
  `.example` files; new clone → `git-crypt unlock` before deploying.
- Image builds contain **no weights** (.dockerignore); prod mounts
  `/srv/veil-core/weights → /app/weights` and syncs via
  `scripts/sync_weights_to_server.sh`.
