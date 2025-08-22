#!/usr/bin/env bash
set -euo pipefail

# Run both visualizers with sensible defaults from repo root or anywhere.
# Usage:
#   bash visualizers/vis_all.sh [CKPT_PATH] [CFG_PATH]
# Defaults:
#   CKPT_PATH = outputs/ckpts/last.pt
#   CFG_PATH  = configs/train.yaml

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CKPT_PATH=${1:-"$REPO_ROOT/outputs/ckpts/last.pt"}
CFG_PATH=${2:-"$REPO_ROOT/configs/train.yaml"}

OUT_DIR="$REPO_ROOT/outputs/vis"
mkdir -p "$OUT_DIR"

echo "[vis_all] Field flow → $OUT_DIR/kmeans_flow.mp4"
PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/visualizers/kmeans_field_vis.py" \
  --ckpt "$CKPT_PATH" \
  --cfg "$CFG_PATH" \
  --out "$OUT_DIR/kmeans_flow.mp4" \
  --n 120 \
  --t_start 0.0 \
  --t_end 1.0 \
  --t_step 0.05 \
  --fps 20

# Also create a GIF (via high-quality MP4→GIF conversion inside the script)
echo "[vis_all] Field flow (GIF) → $OUT_DIR/kmeans_flow.gif"
PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/visualizers/kmeans_field_vis.py" \
  --ckpt "$CKPT_PATH" \
  --cfg "$CFG_PATH" \
  --out "$OUT_DIR/kmeans_flow.gif" \
  --n 120 \
  --t_start 0.0 \
  --t_end 1.0 \
  --t_step 0.05 \
  --fps 20

echo "[vis_all] Particles → $OUT_DIR/particles.gif"
PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/visualizers/kmeans_particles_vis.py" \
  --ckpt "$CKPT_PATH" \
  --out "$OUT_DIR/particles.gif" \
  --num 200 \
  --bounds -4.5 4.5 \
  --steps 50 \
  --fps 20 \
  --size 10 \
  --cfg "$CFG_PATH"

echo "[vis_all] Done"



