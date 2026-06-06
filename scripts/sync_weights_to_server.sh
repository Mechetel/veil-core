#!/usr/bin/env bash
# Sync the local weights/ directory to the production weights volume on the Kamal
# host. Run once after the first deploy, then again whenever weights change.
#
# The server path (default /srv/veil-core/weights) is bind-mounted into the
# container at /app/weights via config/deploy.yml (volumes:). Because it lives on
# the host volume, weights survive image rebuilds and redeploys.
#
# Usage: scripts/sync_weights_to_server.sh deploy@1.2.3.4 [/srv/veil-core/weights]
set -euo pipefail

REMOTE="${1:?usage: sync_weights_to_server.sh user@host [remote_dir]}"
REMOTE_DIR="${2:-/srv/veil-core/weights}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/weights"

echo "Syncing ${LOCAL_DIR}/  ->  ${REMOTE}:${REMOTE_DIR}/"
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
rsync -avz --progress "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
echo "Done."
