# Veil Core — Docker & Deployment Guide

The compute brain of Veil: FastAPI API + two Celery workers (steg / analysis
queues) + Redis. Model weights live **outside the image** on a host volume and
are synced separately, so images stay small and weights update without rebuilds.

The companion Rails app is **veil-web** (see `../veil-web/DEPLOY.md`). Wiring:

```
                  X-Auth-Token: VEIL_CORE_TOKEN
  veil-web ────────────────────────────────────▶ veil-core API (:80 → uvicorn :8000)
  (Rails)                                            │ enqueues via Redis
     ▲                                          steg_worker / analysis_worker
     │            X-Auth-Token: VEIL_CALLBACK_TOKEN     │ reads /app/weights
     └──────────────────────────────────────────────────┘   (host volume)
          POST /callbacks/steganography|steganalysis
```

File map:

| File | Purpose |
|---|---|
| `docker/dev.yml` | local dev stack (compose) |
| `docker/dockerfiles/be.Dockerfile` | dev base image (deps install into a volume at runtime) |
| `docker/dockerfiles/prod.Dockerfile` | production image (deps baked in, weights NOT) |
| `docker/envs/core-dev.env` | non-secret dev wiring (localhost defaults) |
| `docker/secret-envs/*.env` | tokens; `*production.env` + `docker-hub.env` are **git-crypt encrypted** |
| `config/deploy.yml` | Kamal: servers, roles, proxy, Redis accessory, weights volume |
| `.kamal/secrets` | maps env vars → Kamal secrets (values come from the files above) |
| `scripts/sync_weights_to_server.sh` | rsync local `weights/` → server volume |
| `scripts/collect_weights.py` | gather checkpoints from the dissertation repo into `weights/` |

> **git-crypt:** clone on a new machine → `git-crypt unlock <keyfile>` first.
> Kamal reads `docker/secret-envs/*production.env` at deploy time (the ERB header
> in `config/deploy.yml`); with the repo locked those files are ciphertext and
> the deploy fails.

---

## 1. Run locally in Docker (first time ever)

Prereqs: Docker Desktop running; repo git-crypt unlocked; weights present at
`./weights` (if empty: `python scripts/collect_weights.py`).

```bash
cd ~/Projects/veil-core
docker compose -f docker/dev.yml up --build
```

What happens on the first run:
1. The dev image builds (Python 3.12-slim + system libs) — ~1 min.
2. `core` creates a venv in the persisted `core_venv` volume and pip-installs
   `requirements.txt` + latest torch/torchvision (CPU) — **several minutes, once**.
   It then touches `/opt/venv/.ready`.
3. `steg_worker` / `analysis_worker` wait for that sentinel, then start
   (they never pip-install — three writers on one venv volume corrupt it).
4. Redis listens on `localhost:6379`, the API on `http://localhost:8000`
   (uvicorn `--reload`, code is bind-mounted — edits hot-reload).

Verify:

```bash
curl http://localhost:8000/up        # {"status":"ok"}
docker compose -f docker/dev.yml ps  # 4 services running
```

Callbacks go to the web app on your **host** at `http://host.docker.internal:3000`
— start veil-web with `bin/dev` (bare) or its own compose.

### Daily driving

```bash
docker compose -f docker/dev.yml up -d         # start (detached)
docker compose -f docker/dev.yml logs -f core  # tail one service
docker compose -f docker/dev.yml restart steg_worker
docker compose -f docker/dev.yml stop          # stop, keep containers
docker compose -f docker/dev.yml down          # stop + remove containers (venv volume survives)
docker compose -f docker/dev.yml up --build    # after changing be.Dockerfile
docker compose -f docker/dev.yml down -v       # ALSO drop the venv volume (full reset → slow next start)
docker compose -f docker/dev.yml exec core bash
```

---

## 2. One-time server preparation

Any Ubuntu 22.04/24.04 VPS (≥2 GB RAM; torch is hungry) reachable as root by SSH key.

```bash
ssh root@CORE_IP 'curl -fsSL https://get.docker.com | sh'
```

Open ports **22** and **80** (and 443 only if you later add TLS). Nothing else —
Redis is never exposed; it lives on the private `kamal` docker network.

On your **local machine** you need the `kamal` CLI. Easiest (no Gemfile here):

```bash
gem install kamal     # kamal itself depends on dotenv, which deploy.yml's ERB uses
kamal version
```

---

## 3. First deploy — bare IP, no domain

### 3.1 Fill in the placeholders

* `config/deploy.yml` — replace every `CCC.CCC.CCC.CCC` with the **core server IP**
  (3 role entries + the redis accessory host), and set
  `WEB_CALLBACK_URL: http://WEB_IP` (the **web server IP** — port 80, it goes
  through veil-web's kamal-proxy).
* `docker/secret-envs/docker-hub.env` — Docker Hub access token (push scope).
* `docker/secret-envs/veil-core-production.env` — `VEIL_CORE_TOKEN` /
  `VEIL_CALLBACK_TOKEN`; generate with `openssl rand -hex 32`. **The same values
  must be in veil-web's `docker/secret-envs/veil-core-production.env`.**

With no `proxy.hosts` the app answers on **any** hostname → `http://CORE_IP/`
just works. `ssl` stays off (Let's Encrypt needs a domain; see §7 for nip.io).

### 3.2 Ship it

```bash
cd ~/Projects/veil-core
kamal setup
```

`setup` = build the amd64 image (first build is heavy: torch ≈ 1 GB), push to
Docker Hub, install kamal-proxy on the server, boot the **redis** accessory,
then the three roles (`web`, `steg_worker`, `analysis_worker`). The proxy
health-checks `GET /up` on the web role before routing traffic.

### 3.3 Put the weights on the server

The image contains **no weights**; the containers mount
`/srv/veil-core/weights → /app/weights`:

```bash
scripts/sync_weights_to_server.sh root@CORE_IP
ssh root@CORE_IP 'chmod -R a+rX /srv/veil-core/weights'   # containers run as uid 1000
```

Order doesn't matter for correctness — models **lazy-load on first use** — but
until the sync finishes, encode/decode/analyze requests will fail with
"weights not available".

### 3.4 Verify

```bash
curl http://CORE_IP/up          # {"status":"ok"} through kamal-proxy
kamal app logs -r steg_worker   # celery banner, queue "steg"
kamal app logs -r analysis_worker
```

---

## 4. Every next deploy (code changed)

```bash
cd ~/Projects/veil-core
kamal deploy
```

Build → push → rolling replace; the weights volume and Redis are untouched.

```bash
kamal logs                # alias: app logs -f
kamal shell               # alias: bash inside the app container
kamal app details         # what's running where
kamal rollback <version>  # versions: kamal audit
```

---

## 5. Updating weights WITHOUT touching Docker

Weights are deliberately not part of the image — no rebuild, no push:

```bash
# 1. refresh the local weights tree (new/retrained checkpoints)
python scripts/collect_weights.py

# 2. rsync to the server volume
scripts/sync_weights_to_server.sh root@CORE_IP

# 3. restart the containers so the in-memory LRU model caches drop
kamal app boot
```

`kamal app boot` restarts all roles **with the already-deployed image** (seconds,
no build). Restart is required because loaded models are cached in memory
(`lru_cache` in `app/ml/*`); a brand-new weight file that no one has used yet
needs no restart at all.

Adding a **new model key** means editing `app/registry/steg.yaml` /
`analyzers.yaml` — that's code → `kamal deploy` (and sync its weight file).

---

## 6. Stop / start / restart in production

```bash
kamal app stop                      # stop all roles (proxy returns 502)
kamal app start                     # start them again
kamal app boot                      # restart (same image) — also per role:
kamal app boot -r steg_worker
kamal accessory reboot redis        # restart Redis (queued jobs in flight are lost)
kamal accessory logs redis
kamal proxy reboot                  # rarely needed
kamal app remove                    # remove app containers (weights volume survives)
kamal remove                        # nuke everything incl. proxy + accessories (asks to confirm)
```

After a VPS reboot everything self-heals: Kamal runs containers with
`--restart unless-stopped`.

---

## 7. Variant: both apps on ONE server

kamal-proxy allows only **one** host-less (catch-all) app per server. To run
veil-core and veil-web on the same VPS, give each a hostname — no domain needed,
`nip.io` resolves `anything.1.2.3.4.nip.io → 1.2.3.4` for free:

```yaml
# veil-core/config/deploy.yml
proxy:
  app_port: 8000
  healthcheck: { path: /up }
  hosts: [veil-core.CORE_IP.nip.io]

# veil-web/config/deploy.yml
proxy:
  hosts: [veil-web.CORE_IP.nip.io]
```

Then update the cross-references to use those hostnames:
core `WEB_CALLBACK_URL: http://veil-web.CORE_IP.nip.io`,
web `VEIL_CORE_ADDRESS=http://veil-core.CORE_IP.nip.io` (port 80) and
`APP_HOST: veil-web.CORE_IP.nip.io`.

---

## 8. Troubleshooting

| Symptom | Look at / fix |
|---|---|
| `kamal setup` dies at "container not healthy" | `kamal app logs --lines 200`; try `kamal shell` → `curl localhost:8000/up` |
| Encode/analyze fails: model unavailable | weights not synced or unreadable: `ssh root@CORE_IP 'ls -la /srv/veil-core/weights/steg'`; `chmod -R a+rX` |
| Worker idle, jobs stuck queued | `kamal app logs -r steg_worker`; Redis up? `kamal accessory logs redis` |
| Web never receives results | `WEB_CALLBACK_URL` wrong, or `VEIL_CALLBACK_TOKEN` differs between repos |
| Deploy reads garbage env values | repo is git-crypt **locked** — `git-crypt unlock` and retry |
| Server disk filling up | `kamal prune all` (old images/containers) |
| GPU server | rebuild with `--build-arg TORCH_INDEX=cu128` (`builder.args` in deploy.yml), set `CUDA: "true"` in env |
