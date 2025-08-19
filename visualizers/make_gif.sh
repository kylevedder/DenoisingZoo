#!/usr/bin/env bash
set -euo pipefail

# Create an animated GIF from a sequence of PNG frames.
# Defaults:
#   frames: outputs/vis/kmeans_flow_t*.png
#   output: outputs/vis/kmeans_flow.gif
#   fps:    20 (delay = 100/fps centiseconds)
# Usage examples:
#   bash visualizers/make_gif.sh
#   bash visualizers/make_gif.sh --frames "outputs/vis/kmeans_seq_flow_t*.png" --output outputs/vis/anim.gif --fps 30

FRAMES_GLOB="outputs/vis/kmeans_flow_t*.png"
OUTPUT="outputs/vis/kmeans_flow.gif"
FPS=20
METHOD="auto"  # auto | ffmpeg | magick

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frames)
      FRAMES_GLOB="$2"; shift 2;;
    --output)
      OUTPUT="$2"; shift 2;;
    --fps)
      FPS="$2"; shift 2;;
    --method)
      METHOD="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--frames GLOB] [--output PATH] [--fps N]"; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT")"

# Compute ImageMagick delay (centiseconds per frame)
if ! [[ "$FPS" =~ ^[0-9]+$ ]]; then
  echo "--fps must be an integer" >&2; exit 1
fi
DELAY=$((100 / FPS))
if [[ "$DELAY" -lt 1 ]]; then DELAY=1; fi

# Expand frames and verify they exist
shopt -s nullglob
FRAMES=( $FRAMES_GLOB )
shopt -u nullglob
if [[ ${#FRAMES[@]} -eq 0 ]]; then
  echo "No frames matched glob: $FRAMES_GLOB" >&2
  exit 1
fi

echo "Creating GIF: $OUTPUT"
echo "Frames: ${#FRAMES[@]} (delay=${DELAY}cs ~ ${FPS}fps)"

use_ffmpeg=false
use_magick=false

if [[ "$METHOD" == "ffmpeg" ]]; then
  use_ffmpeg=true
elif [[ "$METHOD" == "magick" ]]; then
  use_magick=true
else
  if command -v ffmpeg >/dev/null 2>&1; then
    use_ffmpeg=true
  else
    use_magick=true
  fi
fi

if $use_ffmpeg; then
  # ffmpeg palette method (high quality with good dithering)
  PALETTE="$(dirname "$OUTPUT")/palette.png"
  echo "Method: ffmpeg (palettegen/paletteuse)"
  ffmpeg -y -loglevel error -framerate "$FPS" -pattern_type glob -i "$FRAMES_GLOB" \
    -vf "scale=iw:ih:flags=lanczos,palettegen=stats_mode=full" "$PALETTE"
  ffmpeg -y -loglevel error -framerate "$FPS" -pattern_type glob -i "$FRAMES_GLOB" -i "$PALETTE" \
    -lavfi "scale=iw:ih:flags=lanczos,paletteuse=dither=sierra2_4a" -loop 0 "$OUTPUT"
  rm -f "$PALETTE"
  echo "Done: $OUTPUT"
  exit 0
fi

# Fallback to ImageMagick with enhanced settings to reduce artifacts
if command -v magick >/dev/null 2>&1; then
  IM_CMD=(magick convert)
elif command -v convert >/dev/null 2>&1; then
  IM_CMD=(convert)
else
  echo "ImageMagick not found. Install with: brew install imagemagick" >&2
  exit 1
fi

echo "Method: ImageMagick (enhanced)"
"${IM_CMD[@]}" -delay "$DELAY" -loop 0 ${FRAMES[@]} -coalesce -dither FloydSteinberg -fuzz 1% -colors 256 -layers OptimizeTransparency "$OUTPUT"
echo "Done: $OUTPUT"


