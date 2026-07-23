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
  --ncnn-dir    ncnn CMake package or build/src directory. Default: auto-detect.
  --build-dir   CMake build directory. Default: build
  --case        Example case key. Default: hunyuan_zimu2
EOF
}

fail() {
  echo "quickstart: $*" >&2
  exit 1
}

has_ncnn_package() {
  [[ -f "$1/ncnnConfig.cmake" || -f "$1/ncnn-config.cmake" ]] &&
    [[ -f "$1/ncnn.cmake" ]]
}

ncnn_build_library() {
  local directory="$1"
  local name
  for name in libncnn.a libncnn.so libncnn.dylib ncnn.lib; do
    if [[ -f "$directory/$name" ]]; then
      echo "$directory/$name"
      return 0
    fi
  done
  return 1
}

has_ncnn_build_tree() {
  [[ -f "$1/platform.h" && -f "$1/../../src/net.h" ]] &&
    ncnn_build_library "$1" >/dev/null
}

MODEL=""
NCNN_DIR=""
NCNN_MODE=""
NCNN_LIBRARY_PATH=""
BUILD_DIR="$REPO_ROOT/build"
CASE="hunyuan_zimu2"

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
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

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
  if has_ncnn_package "$NCNN_DIR"; then
    NCNN_MODE="package"
  elif has_ncnn_build_tree "$NCNN_DIR"; then
    NCNN_MODE="build-tree"
  else
    fail "ncnnConfig.cmake was not found and no usable ncnn build tree exists under: $NCNN_DIR"
  fi
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
    if has_ncnn_package "$candidate"; then
      NCNN_DIR="$candidate"
      NCNN_MODE="package"
      break
    fi
    if has_ncnn_build_tree "$candidate"; then
      NCNN_DIR="$candidate"
      NCNN_MODE="build-tree"
      break
    fi
  done
fi

echo "Using model: $MODEL"

CMAKE_ARGS=(-S "$REPO_ROOT" -B "$BUILD_DIR")
if [[ -n "$NCNN_DIR" ]]; then
  NCNN_DIR="$(cd "$NCNN_DIR" && pwd)"
  if [[ "$NCNN_MODE" == "package" ]]; then
    echo "Using ncnn package: $NCNN_DIR"
    CMAKE_ARGS+=("-Dncnn_DIR=$NCNN_DIR")
  else
    NCNN_SOURCE_DIR="$(cd "$NCNN_DIR/../.." && pwd)"
    NCNN_LIBRARY_PATH="$(ncnn_build_library "$NCNN_DIR")"
    echo "Using ncnn build tree: $NCNN_DIR"
    CMAKE_ARGS+=(
      "-DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF"
      "-DNCNN_INCLUDE_DIR=$NCNN_SOURCE_DIR/src"
      "-DNCNN_BUILD_INCLUDE_DIR=$NCNN_DIR"
      "-DNCNN_LIBRARY=$NCNN_LIBRARY_PATH"
    )
  fi
else
  echo "No common ncnn package or build tree found; trying CMake package discovery."
fi

if ! cmake "${CMAKE_ARGS[@]}"; then
  fail "CMake could not find ncnn; pass --ncnn-dir PATH or set ncnn_DIR"
fi
cmake --build "$BUILD_DIR" -j
python "$REPO_ROOT/tools/run_example.py" \
  --model "$MODEL" \
  --binary "$BUILD_DIR/hunyuan_ocr_cli" \
  --case "$CASE"
