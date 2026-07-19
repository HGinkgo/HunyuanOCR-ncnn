#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/quickstart_existing_model.sh [options]

Options:
  --model       Packaged model directory. Default: auto-detect common paths.
  --ncnn-dir    ncnn CMake package directory. Default: auto-detect common paths.
  --build-dir   CMake build directory. Default: build
  --case        Example case key. Default: hf_demo
  --max-tokens  Smoke-test generation limit. Default: 16
EOF
}

fail() {
  echo "quickstart: $*" >&2
  exit 1
}

has_ncnn_config() {
  [[ -f "$1/ncnnConfig.cmake" || -f "$1/ncnn-config.cmake" ]]
}

MODEL=""
NCNN_DIR=""
BUILD_DIR="$REPO_ROOT/build"
CASE="hf_demo"
MAX_TOKENS=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || fail "--model requires a path"
      MODEL="$2"; shift 2 ;;
    --ncnn-dir)
      [[ $# -ge 2 ]] || fail "--ncnn-dir requires a path"
      NCNN_DIR="$2"; shift 2 ;;
    --build-dir)
      [[ $# -ge 2 ]] || fail "--build-dir requires a path"
      BUILD_DIR="$2"; shift 2 ;;
    --case)
      [[ $# -ge 2 ]] || fail "--case requires a value"
      CASE="$2"; shift 2 ;;
    --max-tokens)
      [[ $# -ge 2 ]] || fail "--max-tokens requires an integer"
      MAX_TOKENS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || fail "--max-tokens must be a positive integer"

if [[ -n "$MODEL" ]]; then
  [[ -d "$MODEL" ]] || fail "model directory was not found: $MODEL"
else
  for candidate in \
      "$REPO_ROOT/hunyuan_ocr_ncnn_model" \
      "$REPO_ROOT/assets/hunyuan_ocr_1_5" \
      "$REPO_ROOT/assets/hunyuan_ocr"; do
    if [[ -d "$candidate" ]]; then
      MODEL="$candidate"
      break
    fi
  done
  if [[ -z "$MODEL" ]]; then
    fail "no model directory found; download to $REPO_ROOT/hunyuan_ocr_ncnn_model or pass --model PATH"
  fi
fi
MODEL="$(cd "$MODEL" && pwd)"

if [[ -n "$NCNN_DIR" ]]; then
  has_ncnn_config "$NCNN_DIR" || fail "ncnnConfig.cmake was not found under: $NCNN_DIR"
else
  NCNN_CANDIDATES=()
  if [[ -n "${ncnn_DIR:-}" ]]; then
    NCNN_CANDIDATES+=("$ncnn_DIR")
  fi
  NCNN_CANDIDATES+=(
    "$REPO_ROOT/../ncnn/build/src"
    "/usr/local/lib/cmake/ncnn"
    "/usr/lib/cmake/ncnn"
    "/opt/ncnn/lib/cmake/ncnn"
  )
  if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    IFS=':' read -r -a prefixes <<< "$CMAKE_PREFIX_PATH"
    for prefix in "${prefixes[@]}"; do
      NCNN_CANDIDATES+=("$prefix" "$prefix/lib/cmake/ncnn" "$prefix/share/ncnn")
    done
  fi
  for candidate in "${NCNN_CANDIDATES[@]}"; do
    if has_ncnn_config "$candidate"; then
      NCNN_DIR="$candidate"
      break
    fi
  done
fi

echo "Using model: $MODEL"

CMAKE_ARGS=(-S "$REPO_ROOT" -B "$BUILD_DIR")
if [[ -n "$NCNN_DIR" ]]; then
  NCNN_DIR="$(cd "$NCNN_DIR" && pwd)"
  echo "Using ncnn package: $NCNN_DIR"
  CMAKE_ARGS+=("-Dncnn_DIR=$NCNN_DIR")
else
  echo "No common ncnn package path found; trying CMake package discovery."
fi

if ! cmake "${CMAKE_ARGS[@]}"; then
  fail "CMake could not find ncnn; pass --ncnn-dir PATH or set ncnn_DIR"
fi
cmake --build "$BUILD_DIR" -j
"$SCRIPT_DIR/smoke_test.sh" \
  --model "$MODEL" \
  --binary "$BUILD_DIR/hunyuan_ocr_cli" \
  --case "$CASE" \
  --max-tokens "$MAX_TOKENS"
