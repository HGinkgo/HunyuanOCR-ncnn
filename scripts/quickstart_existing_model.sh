#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/quickstart_existing_model.sh --model ./hunyuan_ocr_ncnn_model --ncnn-dir /path/to/ncnn/lib/cmake/ncnn

Options:
  --model       Packaged HunyuanOCR-ncnn model directory.
  --ncnn-dir    ncnn CMake package directory containing ncnnConfig.cmake.
  --build-dir   CMake build directory. Default: build
  --case        Example case key. Default: hf_demo
EOF
}

MODEL=""
NCNN_DIR=""
BUILD_DIR="$REPO_ROOT/build"
CASE="hf_demo"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --ncnn-dir) NCNN_DIR="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --case) CASE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$NCNN_DIR" ]]; then
  usage
  exit 1
fi

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -Dncnn_DIR="$NCNN_DIR"
cmake --build "$BUILD_DIR" -j
"$SCRIPT_DIR/smoke_test.sh" --model "$MODEL" --binary "$BUILD_DIR/hunyuan_ocr_cli" --case "$CASE"
