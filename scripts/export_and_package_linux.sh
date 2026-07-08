#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_and_package_linux.sh \
    --hf-dir /path/to/HunyuanOCR-hf \
    --pnnx /path/to/pnnx \
    --ncnn-dir /path/to/ncnn/lib/cmake/ncnn \
    --output ./hunyuan_ocr_ncnn_model

Options:
  --hf-dir      HunyuanOCR HuggingFace model directory.
  --pnnx        pnnx executable.
  --ncnn-dir    ncnn CMake package directory containing ncnnConfig.cmake.
  --workspace   Export workspace. Default: current repository directory.
  --output      Packaged runtime model directory. Default: ./hunyuan_ocr_ncnn_model
  --build-dir   CMake build directory. Default: build
  --case        Example case key for the final smoke run. Default: hf_demo
  --copy        Copy artifacts into the package instead of symlinking.
  --skip-run    Do not run the final example.
EOF
}

HF_DIR=""
PNNX=""
NCNN_DIR=""
WORKSPACE="$(pwd)"
OUTPUT="./hunyuan_ocr_ncnn_model"
BUILD_DIR="build"
CASE="hf_demo"
COPY_FLAG=""
SKIP_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-dir) HF_DIR="$2"; shift 2 ;;
    --pnnx) PNNX="$2"; shift 2 ;;
    --ncnn-dir) NCNN_DIR="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --case) CASE="$2"; shift 2 ;;
    --copy) COPY_FLAG="--copy"; shift ;;
    --skip-run) SKIP_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$HF_DIR" || -z "$PNNX" || -z "$NCNN_DIR" ]]; then
  usage
  exit 1
fi

cmake -S . -B "$BUILD_DIR" -Dncnn_DIR="$NCNN_DIR"
cmake --build "$BUILD_DIR" -j

python export/export_all.py \
  --hf-dir "$HF_DIR" \
  --pnnx "$PNNX" \
  --workspace "$WORKSPACE"

python tools/package_model.py \
  --workspace "$WORKSPACE" \
  --output "$OUTPUT" \
  --vision-backend dynamic \
  $COPY_FLAG \
  --force

if [[ "$SKIP_RUN" -eq 0 ]]; then
  python tools/run_example.py --model "$OUTPUT" --case "$CASE"
fi
