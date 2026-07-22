#!/usr/bin/env python3
"""Convert selected pnnx Gemm layers to binary-compatible InnerProduct layers."""

from __future__ import annotations

import argparse
from pathlib import Path


class PatchError(RuntimeError):
    pass


def _params(fields: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for field in fields:
        if "=" not in field:
            raise PatchError(f"invalid ncnn parameter field: {field}")
        key_text, value = field.split("=", 1)
        try:
            key = int(key_text)
        except ValueError as exc:
            raise PatchError(f"invalid ncnn parameter key: {key_text}") from exc
        result[key] = value
    return result


def _require_int(params: dict[int, str], key: int, expected: int, label: str) -> None:
    try:
        actual = int(params.get(key, "0"))
    except ValueError as exc:
        raise PatchError(f"{label} must be an integer") from exc
    if actual != expected:
        raise PatchError(f"compatible Gemm requires {label}={expected}, got {actual}")


def _require_float(params: dict[int, str], key: int, expected: float, label: str) -> None:
    try:
        actual = float(params.get(key, str(expected)))
    except ValueError as exc:
        raise PatchError(f"{label} must be a number") from exc
    if actual != expected:
        raise PatchError(f"compatible Gemm requires {label}={expected:g}, got {actual:g}")


def _convert_line(line: str, requested_name: str) -> str:
    fields = line.split()
    if len(fields) < 6 or fields[0] != "Gemm" or fields[1] != requested_name:
        raise PatchError(f"invalid requested Gemm line: {requested_name}")
    try:
        bottom_count = int(fields[2])
        top_count = int(fields[3])
    except ValueError as exc:
        raise PatchError(f"invalid blob counts for Gemm layer: {requested_name}") from exc
    if bottom_count != 1 or top_count != 1:
        raise PatchError(f"compatible Gemm requires one input and one output: {requested_name}")

    param_start = 4 + bottom_count + top_count
    if len(fields) <= param_start:
        raise PatchError(f"Gemm parameters are missing: {requested_name}")
    params = _params(fields[param_start:])
    _require_float(params, 0, 1.0, "alpha")
    _require_int(params, 2, 0, "transA")
    _require_int(params, 3, 1, "transB")
    _require_int(params, 4, 0, "constantA")
    _require_int(params, 5, 1, "constantB")
    _require_int(params, 6, 1, "constantC")
    _require_int(params, 7, 0, "constantM")
    _require_int(params, 10, -1, "constant_broadcast_type_C")
    _require_int(params, 11, 0, "output_N1M")
    _require_int(params, 12, 0, "output_elempack")
    _require_int(params, 13, 0, "output_elemtype")
    _require_int(params, 14, 0, "output_transpose")
    _require_int(params, 18, 0, "quantize_term")

    try:
        output_size = int(params[8])
        input_size = int(params[9])
    except (KeyError, ValueError) as exc:
        raise PatchError(f"constantN/constantK must be positive integers: {requested_name}") from exc
    if input_size <= 0 or output_size <= 0:
        raise PatchError(f"constantN/constantK must be positive integers: {requested_name}")

    input_blob = fields[4]
    output_blob = fields[5]
    weight_size = input_size * output_size
    return (
        f"InnerProduct {requested_name} 1 1 {input_blob} {output_blob} "
        f"0={output_size} 1=0 2={weight_size} 8=0"
    )


def find_matching_gemm_layers(
    text: str,
    input_size: int,
    output_size: int,
    expected_count: int,
) -> set[str]:
    if input_size <= 0 or output_size <= 0:
        raise PatchError("matching Gemm dimensions must be positive")
    if expected_count <= 0:
        raise PatchError("expected matching Gemm count must be positive")

    matching: set[str] = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[0] != "Gemm":
            continue
        try:
            bottom_count = int(fields[2])
            top_count = int(fields[3])
        except ValueError as exc:
            raise PatchError(f"invalid blob counts for Gemm layer: {fields[1]}") from exc
        param_start = 4 + bottom_count + top_count
        params = _params(fields[param_start:])
        try:
            layer_output_size = int(params[8])
            layer_input_size = int(params[9])
        except (KeyError, ValueError) as exc:
            raise PatchError(
                f"constantN/constantK must be positive integers: {fields[1]}"
            ) from exc
        if layer_input_size != input_size or layer_output_size != output_size:
            continue
        if fields[1] in matching:
            raise PatchError(f"matching Gemm layer appears more than once: {fields[1]}")
        _convert_line(line, fields[1])
        matching.add(fields[1])

    if len(matching) != expected_count:
        raise PatchError(
            f"expected {expected_count} matching Gemm layers, found {len(matching)}"
        )
    return matching


def patch_param_text(text: str, requested_layers: set[str]) -> str:
    if not requested_layers:
        raise PatchError("at least one Gemm layer must be requested")
    found: set[str] = set()
    output: list[str] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "Gemm" and fields[1] in requested_layers:
            name = fields[1]
            if name in found:
                raise PatchError(f"requested Gemm layer appears more than once: {name}")
            output.append(_convert_line(line, name))
            found.add(name)
        else:
            output.append(line)
    missing = requested_layers - found
    if missing:
        raise PatchError("requested Gemm layer not found: " + ", ".join(sorted(missing)))
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param", type=Path, required=True, help="source ncnn param")
    parser.add_argument("--output", type=Path, required=True, help="patched ncnn param")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--layer", action="append", help="Gemm layer name")
    selector.add_argument(
        "--matching-shape",
        type=int,
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="convert every compatible Gemm with this input/output size",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="required guard for --matching-shape",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = args.param.read_text(encoding="utf-8")
        if args.matching_shape:
            if args.expected_count is None:
                raise PatchError("--matching-shape requires --expected-count")
            requested_layers = find_matching_gemm_layers(
                source,
                input_size=args.matching_shape[0],
                output_size=args.matching_shape[1],
                expected_count=args.expected_count,
            )
        else:
            if args.expected_count is not None:
                raise PatchError("--expected-count requires --matching-shape")
            requested_layers = set(args.layer)
        patched = patch_param_text(source, requested_layers)
        args.output.write_text(patched, encoding="utf-8")
    except (OSError, PatchError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"patched_param={args.output}")
    print("converted_layers=" + ",".join(sorted(requested_layers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
