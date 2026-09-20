#!/usr/bin/env python3
"""Validate a HY4 MXFP4 experts + FP8 UE8M0 mixed checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.hy4_mxfp4_rtn_to_fp8_ue8m0 import (  # noqa: E402
    sparse_main_layers,
    validate_rtn_stage1,
    validate_output,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON top level is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shards(
    output: Path,
    weight_map: dict[str, str],
    full_duplicate_check: bool,
) -> tuple[Counter[str], int]:
    indexed_shards = set(weight_map.values())
    disk_paths = sorted(output.glob("*.safetensors"))
    disk_shards = {path.name for path in disk_paths}
    if indexed_shards != disk_shards:
        raise RuntimeError(
            "disk/index shard mismatch: "
            f"missing={sorted(indexed_shards - disk_shards)[:10]}, "
            f"unreferenced={sorted(disk_shards - indexed_shards)[:10]}"
        )

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        expected_by_shard[shard].add(key)

    physical_owners: dict[str, list[str]] = defaultdict(list)
    inode_owners: dict[tuple[int, int], list[str]] = defaultdict(list)
    size_groups: dict[int, list[Path]] = defaultdict(list)
    dtype_counts: Counter[str] = Counter()

    for path in disk_paths:
        if path.is_symlink():
            raise RuntimeError(f"symlink shard is not allowed: {path.name}")
        stat = path.stat()
        if stat.st_nlink != 1:
            raise RuntimeError(f"hard-linked shard is not allowed: {path.name}")
        inode_owners[(stat.st_dev, stat.st_ino)].append(path.name)
        size_groups[stat.st_size].append(path)

        with safe_open(path, framework="pt", device="cpu") as reader:
            keys = set(reader.keys())
            if keys != expected_by_shard[path.name]:
                raise RuntimeError(f"stored/indexed key mismatch: {path.name}")
            for key in keys:
                physical_owners[key].append(path.name)
                dtype_counts[str(reader.get_slice(key).get_dtype())] += 1

    duplicate_keys = {key: owners for key, owners in physical_owners.items() if len(owners) != 1}
    if duplicate_keys:
        raise RuntimeError(
            f"duplicate tensor keys across shards: {list(duplicate_keys.items())[:10]}"
        )
    if set(physical_owners) != set(weight_map):
        raise RuntimeError("physical tensor keys do not exactly match the index")

    duplicate_inodes = [owners for owners in inode_owners.values() if len(owners) > 1]
    if duplicate_inodes:
        raise RuntimeError(f"duplicate shard inodes: {duplicate_inodes[:10]}")

    same_size_candidates = [group for group in size_groups.values() if len(group) > 1]
    if full_duplicate_check:
        digest_groups: dict[tuple[int, str], list[str]] = defaultdict(list)
        for group in same_size_candidates:
            for path in group:
                digest_groups[(path.stat().st_size, sha256(path))].append(path.name)
        content_duplicates = [names for names in digest_groups.values() if len(names) > 1]
        if content_duplicates:
            raise RuntimeError(f"byte-identical shard files: {content_duplicates[:10]}")
        print(
            "[audit] full SHA256 duplicate-shard check: PASS "
            f"(candidates={sum(map(len, same_size_candidates))})"
        )
    else:
        print(
            "[audit] duplicate shard key/inode/link check: PASS "
            f"(same-size candidates={sum(map(len, same_size_candidates))}; "
            "use --full-duplicate-check for SHA256)"
        )
    return dtype_counts, len(indexed_shards)


def validate_mxfp4_samples(
    stage1: Path,
    output: Path,
    output_weight_map: dict[str, str],
    sample_count: int,
) -> int:
    stage1_weight_map = load_json(stage1 / "model.safetensors.index.json")["weight_map"]
    mxfp4_keys = sorted(
        key
        for key in output_weight_map
        if key.endswith((".weight", ".weight_scale"))
        and ".mlp.experts." in key
        and key in stage1_weight_map
    )
    actual_count = min(sample_count, len(mxfp4_keys))
    if actual_count == 0:
        raise RuntimeError("no MXFP4 tensors available for exact-match sampling")
    sample_indices = {
        round(index * (len(mxfp4_keys) - 1) / max(actual_count - 1, 1))
        for index in range(actual_count)
    }
    for sample_index in sorted(sample_indices):
        key = mxfp4_keys[sample_index]
        with safe_open(
            stage1 / stage1_weight_map[key],
            framework="pt",
            device="cpu",
        ) as source_reader:
            source_tensor = source_reader.get_tensor(key)
        with safe_open(
            output / output_weight_map[key],
            framework="pt",
            device="cpu",
        ) as output_reader:
            output_tensor = output_reader.get_tensor(key)
        if not torch.equal(source_tensor, output_tensor):
            raise RuntimeError(f"MXFP4 tensor changed during Stage 2: {key}")
    return len(sample_indices)


def validate_config(
    output: Path,
    bf16: Path,
) -> None:
    bf16_config = load_json(bf16 / "config.json")
    output_config = load_json(output / "config.json")
    for field in ("model_type", "num_key_value_heads", "layer_types", "use_cache"):
        if output_config.get(field) != bf16_config.get(field):
            raise RuntimeError(f"config field is not synchronized: {field}")

    quantization_config = output_config.get("quantization_config", {})
    expected_quantization = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0",
    }
    for field, expected in expected_quantization.items():
        if quantization_config.get(field) != expected:
            raise RuntimeError(
                f"invalid quantization_config.{field}: "
                f"{quantization_config.get(field)!r} != {expected!r}"
            )

    private_fields = [
        field
        for field in output_config
        if "angelslim" in field.lower() or field in {"mtp_quant_algo", "mtp_quant_method"}
    ]
    if private_fields:
        raise RuntimeError(f"private fields remain in config.json: {private_fields}")

    sidecar = load_json(output / "angelslim_config.json").get("rtn_config", {})
    if sidecar.get("algorithm") != "rtn":
        raise RuntimeError("angelslim_config.json does not declare RTN")
    if sidecar.get("mtp_quant_algo") != "FP8":
        raise RuntimeError("angelslim_config.json does not declare MTP FP8")
    sidecar_keys = load_json(output / "angelslim_config.json")
    unexpected_fields = [field for field in sidecar_keys if field != "rtn_config"]
    if unexpected_fields:
        raise RuntimeError(f"unexpected non-RTN sidecar fields: {unexpected_fields}")
    if not (output / "hf_quant_config.json.mxfp4").is_file():
        raise RuntimeError("missing hf_quant_config.json.mxfp4")

    backups = sorted(output.glob("config.json.bak*")) + sorted(
        output.glob("angelslim_config.json.bak*")
    )
    if backups:
        raise RuntimeError(f"unexpected config backups: {backups}")


def main(
    stage1_path: str,
    output_path: str,
    bf16_path: str,
    full_duplicate_check: bool = False,
    sample_count: int = 72,
) -> None:
    stage1 = Path(stage1_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    bf16 = Path(bf16_path).expanduser().resolve()
    validate_rtn_stage1(str(stage1))

    stage1_config = load_json(stage1 / "config.json")
    sparse_layers = sparse_main_layers(stage1_config)
    num_experts = int(stage1_config["n_routed_experts"])
    validate_output(str(output), sparse_layers, num_experts)
    print("[audit] quantized tensor layout: PASS")

    index = load_json(output / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    dtype_counts, shard_count = validate_shards(
        output,
        weight_map,
        full_duplicate_check,
    )
    if dtype_counts.get("F8_E8M0", 0):
        raise RuntimeError(
            "RTN MXFP4 weight_scale tensors must be stored as U8 raw E8M0 "
            "bytes; F8_E8M0 is incompatible with the current HYV4 loader"
        )
    exact_samples = validate_mxfp4_samples(
        stage1,
        output,
        weight_map,
        sample_count,
    )
    validate_config(output, bf16)
    print("[audit] expert policy: data-free RTN")

    print(
        "[audit] shards={shards}, tensors={tensors}, MXFP4={mxfp4}, "
        "FP8={fp8}, dtypes={dtypes}".format(
            shards=shard_count,
            tensors=len(weight_map),
            mxfp4=sum(key.endswith(".weight_scale") for key in weight_map),
            fp8=sum(key.endswith(".scale") for key in weight_map),
            dtypes=dict(sorted(dtype_counts.items())),
        )
    )
    print(f"[audit] representative MXFP4 exact-match samples={exact_samples}")
    print("[audit] index/key/link/config/sidecar checks: PASS")
    print("FINAL_AUDIT=PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--bf16-path", required=True)
    parser.add_argument("--sample-count", type=int, default=72)
    parser.add_argument("--full-duplicate-check", action="store_true")
    args = parser.parse_args()
    main(
        stage1_path=args.stage1_path,
        output_path=args.output_path,
        bf16_path=args.bf16_path,
        full_duplicate_check=args.full_duplicate_check,
        sample_count=args.sample_count,
    )
