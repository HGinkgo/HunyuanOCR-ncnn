#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/smoke_test.sh --model ./hunyuan_ocr_ncnn_model

Options:
  --model      Packaged HunyuanOCR-ncnn model directory.
  --binary     Path to hunyuan_ocr_cli. Default: build/hunyuan_ocr_cli
  --case       Example case key. Default: hf_demo
  --max-tokens Optional generation token limit.
EOF
}

MODEL=""
BINARY="build/hunyuan_ocr_cli"
CASE="hf_demo"
MAX_TOKENS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --case) CASE="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS=(--max-tokens "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  usage
  exit 1
fi

python tools/run_example.py \
  --model "$MODEL" \
  --binary "$BINARY" \
  --case "$CASE" \
  "${MAX_TOKENS[@]}"
