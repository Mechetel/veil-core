#!/usr/bin/env bash
# Vendor the steganogan/ and steganalyzers/ Python packages from the dissertation
# repo into veil-core's import root. Required because .steg checkpoints are pickled
# SteganoGAN objects that unpickle under the original `steganogan.*` module path.
#
# Code only — training runs, datasets and model dumps are excluded (weights are
# handled separately by collect_weights.py).
set -euo pipefail

SRC="${1:-$HOME/Projects/phd_dissertation/state_3/Attention-Steganogan}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Vendoring steganogan/ and steganalyzers/ from: $SRC"
echo "                                          into: $DEST"

common_excludes=(
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
)

rsync -a --delete "${common_excludes[@]}" \
  "$SRC/steganogan/" "$DEST/steganogan/"

rsync -a --delete "${common_excludes[@]}" \
  --exclude 'runs/' \
  --exclude 'runs_alaska/' \
  --exclude 'data/alaska2-image-steganalysis/' \
  "$SRC/steganalyzers/" "$DEST/steganalyzers/"

echo "Done. Vendored packages:"
echo "  $DEST/steganogan"
echo "  $DEST/steganalyzers"
