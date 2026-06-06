# Veil Core

FastAPI compute service for **Veil** — the steganography brain that wraps the
Attention-SteganoGAN neural networks. Consumed by the `veil-web` Rails UI over
HTTP (ActiveResource). This is the foundation: a token-protected hello endpoint
plus a Docker/Kamal deploy setup. Real encode / decode / steganalysis endpoints
come later.

## API

| Method | Path  | Auth          | Response              |
|--------|-------|---------------|-----------------------|
| GET    | `/up` | none          | `{"status":"ok"}`     |
| GET    | `/`   | `X-Auth-Token`| `{"message":"hello"}` |

Auth: every protected route requires the `X-Auth-Token` header to equal
`VEIL_CORE_TOKEN`. This is the same secret `veil-web` sends.

## Run

### 1. Bare (venv)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
VEIL_CORE_TOKEN=dev-veil-token .venv/bin/uvicorn app.main:app --reload --port 8000
```

### 2. Docker Compose (dev)

```bash
docker compose -f docker/dev.yml up
```

Reads `VEIL_CORE_TOKEN` from `docker/secret-envs/veil-core.env`.

### 3. Kamal (prod)

Set the real token in `docker/secret-envs/veil-core-production.env` and the
registry password in `docker/secret-envs/docker-hub.env`, point `servers.web` in
`config/deploy.yml` at your host, then:

```bash
gem install kamal   # if not already installed
kamal setup
```

## Smoke test

```bash
curl -s http://localhost:8000/up                            # {"status":"ok"}
curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/   # 401
curl -s -H 'X-Auth-Token: dev-veil-token' http://localhost:8000/ # {"message":"hello"}
```
