#!/usr/bin/env python3
"""Patch pnnx text decoder ncnn param to expose SDPA KV cache inputs/outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add KV cache blobs to text_decoder_kv.ncnn.param.")
    parser.add_argument("--param", type=Path, required=True, help="text_decoder_kv.ncnn.param to patch in place.")
    parser.add_argument("--layers", type=int, default=24, help="Expected SDPA layer count. Default: 24")
    parser.add_argument("--head-dim", type=int, default=128, help="Attention head dimension. Default: 128")
    return parser.parse_args()


def noutputs(line: str) -> int:
    fields = line.split()
    return int(fields[3]) if len(fields) >= 4 else 0


def main() -> int:
    args = parse_args()
    scale = f"{1.0 / math.sqrt(args.head_dim):g}"
    raw = args.param.read_text(encoding="utf-8")
    if "out_cache_k0" in raw:
        raise SystemExit(f"{args.param} already contains KV cache outputs")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "7767517":
        raise SystemExit(f"not an ncnn param file: {args.param}")

    backup = args.param.with_suffix(args.param.suffix + ".nokv")
    backup.write_text(raw, encoding="utf-8")
    body = lines[2:]
    input_insert_idx = None
    for i, line in enumerate(body):
        fields = line.split()
        if len(fields) >= 5 and fields[0] == "Input" and fields[1].startswith("in"):
            input_insert_idx = i + 1
    if input_insert_idx is None:
        raise SystemExit("could not find Input in*")

    sdpa_index = 0
    out_body: list[str] = []
    for line in body:
        fields = line.split()
        if fields and fields[0] == "SDPA":
            input_count = int(fields[2])
            output_count = int(fields[3])
            blobs = fields[4 : 4 + input_count + output_count]
            inputs = blobs[:input_count]
            outputs = blobs[input_count:]
            params = fields[4 + input_count + output_count :]
            param_map = dict(item.split("=", 1) for item in params)
            param_map["5"] = "1"
            param_map["6"] = scale
            param_map["7"] = "1"
            new_inputs = inputs + [f"cache_k{sdpa_index}", f"cache_v{sdpa_index}"]
            new_outputs = outputs + [f"out_cache_k{sdpa_index}", f"out_cache_v{sdpa_index}"]
            param_text = " ".join(f"{key}={param_map[key]}" for key in sorted(param_map, key=lambda x: int(x)))
            out_body.append(
                f"{fields[0]:<24} {fields[1]:<24} {len(new_inputs)} {len(new_outputs)} "
                f"{' '.join(new_inputs + new_outputs)} {param_text}"
            )
            sdpa_index += 1
        else:
            out_body.append(line)
    if sdpa_index != args.layers:
        raise SystemExit(f"expected {args.layers} SDPA layers, got {sdpa_index}")

    cache_blob_names: list[str] = []
    for i in range(sdpa_index):
        cache_blob_names.extend([f"cache_k{i}", f"cache_v{i}"])
    out_body.insert(
        input_insert_idx,
        f"{'Input':<24} {'kv_cache':<24} 0 {len(cache_blob_names)} {' '.join(cache_blob_names)}",
    )
    layer_lines = [line for line in out_body if line.strip()]
    args.param.write_text(
        "\n".join([lines[0], f"{len(layer_lines)} {sum(noutputs(line) for line in layer_lines)}"] + out_body) + "\n",
        encoding="utf-8",
    )
    print("backup:", backup)
    print("sdpa_rewritten:", sdpa_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
