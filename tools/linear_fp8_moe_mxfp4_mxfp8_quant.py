# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HY4 single-pass mixed-precision quantiser: MXFP4 routed experts + FP8 UE8M0 dense.

Input: a BF16 HY4 HuggingFace checkpoint with batched expert tensors::

    model.layers.{L}.mlp.experts.gate_up_proj   (E, 2*I, H)  bf16
    model.layers.{L}.mlp.experts.down_proj       (E, H,   I)  bf16

Output: a single checkpoint where:

    * Main routed experts (sparse layers) -> MXFP4 RTN (E2M1 values packed
      two per uint8, E8M0 uint8 scale per 32 values), split into per-expert
      gate_proj / up_proj / down_proj keys.
    * MTP routed experts (fused 3-D) -> FP8 E4M3 + UE8M0 128x128 blockwise,
      split into per-expert keys.
    * Dense linear weights (attention, shared experts, dense MLP, indexer
      wk/wq_b) -> FP8 E4M3 + UE8M0 128x128 blockwise.
    * Everything else (lm_head, embed_tokens, router gate, norms, hc_*,
      learnable_sink, eh_proj, indexer weights_proj, biases) -> passthrough.

Usage::

    python tools/linear_fp8_moe_mxfp4_mxfp8_quant.py \\
        --input-path /path/to/hy4-bf16 \\
        --output-path /path/to/hy4-mxfp4-fp8 \\
        [--num-workers 8] [--use-gpu] [--skip-validate]
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MXFP4_GROUP_SIZE = 32
MXFP4_E8M0_BIAS = 127
MXFP4_E2M1_MAX = 6.0
MXFP4_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

FP8_UE8M0_BLOCK_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_MIN = float(torch.finfo(FP8_DTYPE).min)

INDEX_NAME = "model.safetensors.index.json"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_FUSED_EXPERT_RE = re.compile(
    r"^model\.(?P<prefix>layers|mtp_layers)\.(?P<layer>\d+)"
    r"\.mlp\.experts\.(?P<projection>gate_up_proj|down_proj)$"
)

_FP8_WEIGHT_RES = [
    re.compile(
        r"^model\.(?:layers|mtp_layers)\.\d+\.self_attn\."
        r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|linear_gate)"
        r"\.weight$"
    ),
    re.compile(
        r"^model\.(?:layers|mtp_layers)\.\d+\.self_attn\.indexer\." r"(?:wk|wq_b)\.weight$"
    ),
    re.compile(
        r"^model\.(?:layers|mtp_layers)\.\d+\.mlp\." r"(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
    re.compile(
        r"^model\.(?:layers|mtp_layers)\.\d+\.mlp\.shared_experts\."
        r"(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
]

# ---------------------------------------------------------------------------
# MXFP4 RTN quantisation
# ---------------------------------------------------------------------------


def _mxfp4_cast_to_e2m1(weight):
    bounds = MXFP4_E2M1_BOUNDS.to(weight.device)
    tie_round_up_mask = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8, device=weight.device
    )
    tie_round_up_mask = tie_round_up_mask.expand(*weight.shape, 7)
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ordinal = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    round_up = torch.any((weight_abs.unsqueeze(-1) == bounds) * tie_round_up_mask, dim=-1)
    return (sign_bit * 0b1000 + ordinal + round_up).to(torch.uint8)


@torch.no_grad()
def mxfp4_rtn_pack(weight, group_size=MXFP4_GROUP_SIZE):
    if weight.ndim != 2:
        raise ValueError(f"MXFP4 expects 2-D weight, got shape {tuple(weight.shape)}")
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"in_features={weight.shape[-1]} not divisible by group_size={group_size}"
        )

    blocks = weight.float().reshape(weight.shape[0], -1, group_size)
    block_amax = blocks.abs().amax(dim=-1)
    scale_target = (block_amax / MXFP4_E2M1_MAX).clamp_min(1e-30)
    decoded_scale = torch.pow(2.0, torch.ceil(torch.log2(scale_target)))
    decoded_scale = torch.where(block_amax == 0, torch.ones_like(decoded_scale), decoded_scale)

    scaled = (blocks / decoded_scale.unsqueeze(-1)).reshape(weight.shape)
    codes = _mxfp4_cast_to_e2m1(scaled)
    packed_weight = ((codes[:, 1::2] << 4) | codes[:, 0::2]).contiguous()
    encoded_scale = (torch.log2(decoded_scale.float()) + MXFP4_E8M0_BIAS).to(torch.uint8)
    return packed_weight.cpu(), encoded_scale.cpu()


# ---------------------------------------------------------------------------
# FP8 UE8M0 128x128 blockwise quantisation
# ---------------------------------------------------------------------------


@torch.no_grad()
def fp8_ue8m0_quantize(weight, block_size=FP8_UE8M0_BLOCK_SIZE):
    if weight.ndim != 2:
        raise ValueError(f"FP8 expects 2-D weight, got shape {tuple(weight.shape)}")

    rows, columns = weight.shape
    row_blocks = (rows + block_size - 1) // block_size
    column_blocks = (columns + block_size - 1) // block_size
    padded_rows = row_blocks * block_size
    padded_columns = column_blocks * block_size
    source = weight.float()
    if padded_rows != rows or padded_columns != columns:
        source = torch.nn.functional.pad(
            source, (0, padded_columns - columns, 0, padded_rows - rows)
        )

    blocks = source.reshape(row_blocks, block_size, column_blocks, block_size)
    block_amax = blocks.abs().amax(dim=(1, 3))
    safe_amax = torch.where(block_amax > 0, block_amax, torch.ones_like(block_amax))
    exponent = torch.ceil(torch.log2(safe_amax / FP8_MAX))
    encoded_scale = (exponent + MXFP4_E8M0_BIAS).clamp_(0, 255).to(torch.uint8)
    encoded_scale = torch.where(
        block_amax > 0,
        encoded_scale,
        torch.full_like(encoded_scale, MXFP4_E8M0_BIAS),
    )
    decoded_scale = torch.pow(2.0, encoded_scale.to(torch.int32).float() - MXFP4_E8M0_BIAS)
    quantized = (
        (blocks / decoded_scale[:, None, :, None])
        .clamp_(FP8_MIN, FP8_MAX)
        .reshape(padded_rows, padded_columns)[:rows, :columns]
        .to(FP8_DTYPE)
        .contiguous()
    )
    return quantized, encoded_scale.contiguous()


# ---------------------------------------------------------------------------
# Layer classification helpers
# ---------------------------------------------------------------------------


def sparse_main_layers(config):
    num_hidden_layers = int(config["num_hidden_layers"])
    layer_types = config.get("mlp_layer_types")
    if isinstance(layer_types, list) and len(layer_types) >= num_hidden_layers:
        layers = [
            i
            for i, lt in enumerate(layer_types[:num_hidden_layers])
            if str(lt).lower() in {"sparse", "moe"}
        ]
    else:
        layers = list(range(int(config.get("first_k_dense_replace", 1)), num_hidden_layers))
    if not layers:
        raise ValueError("No sparse/MoE main-model layers found in config.json")
    return layers


def _is_fp8_weight(name):
    return any(p.fullmatch(name) for p in _FP8_WEIGHT_RES)


def _passthrough_module_name(key):
    for suffix in (".weight", ".bias"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


# ---------------------------------------------------------------------------
# Per-shard worker
# ---------------------------------------------------------------------------


def _worker_device(use_gpu):
    if not use_gpu or not torch.cuda.is_available():
        return "cpu"
    identity = mp.current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_idx = worker_idx % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    return f"cuda:{device_idx}"


def _store_fp8(state, weight_map, output_name, key, weight, device):
    source = weight.to(device)
    quantized, scale = fp8_ue8m0_quantize(source)
    state[key] = quantized.cpu()
    scale_key = key[: -len(".weight")] + ".scale"
    state[scale_key] = scale.cpu()
    weight_map[key] = output_name
    weight_map[scale_key] = output_name
    del source, quantized, scale


def _store_mxfp4(state, weight_map, output_name, module_name, weight, device):
    packed, scale = mxfp4_rtn_pack(weight.to(device))
    w_key = f"{module_name}.weight"
    s_key = f"{module_name}.weight_scale"
    state[w_key] = packed
    state[s_key] = scale
    weight_map[w_key] = output_name
    weight_map[s_key] = output_name
    del packed, scale


def process_shard(task):
    (
        shard_name,
        input_path,
        output_path,
        manifest_dir,
        sparse_layer_set,
        use_gpu,
    ) = task
    sparse_layers = set(sparse_layer_set)
    output_name = shard_name
    device = _worker_device(use_gpu)

    state = {}
    weight_map = {}
    ignored_modules = set()
    mxfp4_modules = 0
    fp8_modules = 0

    with safe_open(os.path.join(input_path, shard_name), framework="pt", device="cpu") as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)

            # --- Case A: fused batched experts ---
            match = _FUSED_EXPERT_RE.fullmatch(key)
            if match is not None and tensor.ndim == 3:
                prefix_type = match.group("prefix")  # "layers" or "mtp_layers"
                layer_idx = int(match.group("layer"))
                projection = match.group("projection")
                is_main_sparse = prefix_type == "layers" and layer_idx in sparse_layers

                expert_prefix = key.rsplit(".experts.", 1)[0] + ".experts"
                for expert_idx in range(tensor.shape[0]):
                    if projection == "gate_up_proj":
                        gate, up = tensor[expert_idx].chunk(2, dim=0)
                        projections = [("gate_proj", gate), ("up_proj", up)]
                    else:
                        projections = [("down_proj", tensor[expert_idx])]

                    for proj_name, expert_weight in projections:
                        module_name = f"{expert_prefix}.{expert_idx}.{proj_name}"
                        if is_main_sparse:
                            _store_mxfp4(
                                state,
                                weight_map,
                                output_name,
                                module_name,
                                expert_weight,
                                device,
                            )
                            mxfp4_modules += 1
                        else:
                            target_key = f"{module_name}.weight"
                            _store_fp8(
                                state,
                                weight_map,
                                output_name,
                                target_key,
                                expert_weight,
                                device,
                            )
                            fp8_modules += 1
                del tensor
                if device != "cpu":
                    torch.cuda.empty_cache()
                continue

            # --- Case B: regular linear -> FP8 ---
            if _is_fp8_weight(key):
                _store_fp8(state, weight_map, output_name, key, tensor, device)
                fp8_modules += 1
                del tensor
                if device != "cpu":
                    torch.cuda.empty_cache()
                continue

            # --- Case C: passthrough ---
            state[key] = tensor
            weight_map[key] = output_name
            ignored_modules.add(_passthrough_module_name(key))

    save_file(state, os.path.join(output_path, output_name), metadata={"format": "pt"})

    manifest = {
        "weight_map": weight_map,
        "total_size": sum(t.numel() * t.element_size() for t in state.values()),
        "mxfp4_modules": mxfp4_modules,
        "fp8_modules": fp8_modules,
        "ignored_modules": sorted(ignored_modules),
    }
    manifest_name = shard_name + ".json"
    _atomic_write_json(manifest, os.path.join(manifest_dir, manifest_name))
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest_name


# ---------------------------------------------------------------------------
# Multi-process orchestration
# ---------------------------------------------------------------------------


def _run_tasks(tasks, workers):
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        return [process_shard(t) for t in tqdm(tasks, desc="MXFP4+FP8")]
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        return list(
            tqdm(
                pool.imap_unordered(process_shard, tasks),
                total=len(tasks),
                desc="MXFP4+FP8",
            )
        )


def _merge_manifests(manifest_dir, names):
    weight_map = {}
    total_size = 0
    mxfp4_total = 0
    fp8_total = 0
    ignored_modules = set()
    for name in names:
        manifest = _load_json(os.path.join(manifest_dir, name))
        overlap = set(weight_map).intersection(manifest["weight_map"])
        if overlap:
            raise RuntimeError(f"Duplicate output tensors: {sorted(overlap)[:20]}")
        weight_map.update(manifest["weight_map"])
        total_size += int(manifest["total_size"])
        mxfp4_total += int(manifest.get("mxfp4_modules", 0))
        fp8_total += int(manifest.get("fp8_modules", 0))
        ignored_modules.update(manifest["ignored_modules"])
    return weight_map, total_size, mxfp4_total, fp8_total, sorted(ignored_modules)


# ---------------------------------------------------------------------------
# JSON / file utilities
# ---------------------------------------------------------------------------


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(value, path):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _collect_shards(input_path):
    index_path = os.path.join(input_path, INDEX_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    if os.path.isfile(os.path.join(input_path, "model.safetensors")):
        return ["model.safetensors"]
    raise FileNotFoundError(f"No safetensors checkpoint found under {input_path}")


def _copy_auxiliary_files(input_path, output_path):
    generated = {
        INDEX_NAME,
        "config.json",
        "angelslim_config.json",
        "hf_quant_config.json",
    }
    for name in os.listdir(input_path):
        src = os.path.join(input_path, name)
        dst = os.path.join(output_path, name)
        if name in generated or name.endswith(".safetensors"):
            continue
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _build_fp8_config(ignored_modules):
    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [FP8_UE8M0_BLOCK_SIZE, FP8_UE8M0_BLOCK_SIZE],
        "modules_to_not_convert": ignored_modules,
        "scale_fmt": "ue8m0",
    }


def _build_mxfp4_sidecar(source_config, sparse_layers):
    first_moe = min(sparse_layers)
    num_hidden_layers = int(source_config["num_hidden_layers"])
    exclude = ["lm_head"]
    for layer in range(first_moe):
        exclude.append(f"model.layers.{layer}*")
    for layer in sparse_layers:
        exclude.extend(
            [
                f"model.layers.{layer}.self_attn*",
                f"model.layers.{layer}.mlp.shared_experts*",
                f"model.layers.{layer}.mlp.gate*",
                f"model.layers.{layer}.hc_*",
                f"model.layers.{layer}.input_layernorm*",
                f"model.layers.{layer}.post_attention_layernorm*",
            ]
        )
    exclude.append("model.mtp_layers*")
    return {
        "producer": {"name": "AngelSlim", "tool": Path(__file__).name},
        "quantization": {
            "quant_algo": "MXFP4",
            "kv_cache_quant_algo": None,
            "group_size": MXFP4_GROUP_SIZE,
            "exclude_modules": exclude,
            "num_hidden_layers": num_hidden_layers,
        },
    }


def _build_angelslim_sidecar(source_config, sparse_layers, mxfp4_count, fp8_count):
    return {
        "rtn_config": {
            "algorithm": "rtn",
            "weight_format": "mxfp4",
            "scale_format": "ue8m0",
            "group_size": MXFP4_GROUP_SIZE,
            "mxfp4_main_expert_layers": sparse_layers,
            "mxfp4_main_expert_module_count": mxfp4_count,
            "fp8_ue8m0_module_count": fp8_count,
            "mtp_quant_algo": "FP8",
            "mtp_quant_method": "rtn",
        }
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_MXFP4_TENSOR_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale)$"
)


def validate_output(output_path, sparse_layers, num_experts):
    index = _load_json(os.path.join(output_path, INDEX_NAME))
    weight_map = index["weight_map"]
    keys = set(weight_map)

    by_shard = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    metadata = {}
    for shard, shard_keys in by_shard.items():
        path = os.path.join(output_path, shard)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing output shard: {shard}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(shard_keys):
                raise RuntimeError(f"Stored/indexed key mismatch in {shard}")
            for key in shard_keys:
                t = reader.get_slice(key)
                metadata[key] = (str(t.get_dtype()), tuple(t.get_shape()))

    missing = []
    for layer in sparse_layers:
        for expert in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                for suffix in (".weight", ".weight_scale"):
                    if base + suffix not in keys:
                        missing.append(base + suffix)
    if missing:
        raise RuntimeError(f"Missing MXFP4 tensors: {missing[:20]}")

    mxfp4_count = 0
    fp8_count = 0
    errors = []
    for key in keys:
        if key.endswith(".weight_scale"):
            m = _MXFP4_TENSOR_RE.fullmatch(key)
            if m is None:
                errors.append((key, "unexpected weight_scale"))
                continue
            base = key[: -len(".weight_scale")]
            w_key = base + ".weight"
            w_dtype, w_shape = metadata[w_key]
            s_dtype, s_shape = metadata[key]
            expected = (w_shape[0], w_shape[1] * 2 // MXFP4_GROUP_SIZE)
            if w_dtype != "U8" or s_dtype != "U8" or s_shape != expected:
                errors.append((base, metadata[w_key], metadata[key], expected))
            mxfp4_count += 1
        elif key.endswith(".scale"):
            base = key[: -len(".scale")]
            w_key = base + ".weight"
            if w_key not in keys:
                errors.append((key, "missing weight"))
                continue
            w_dtype, w_shape = metadata[w_key]
            s_dtype, s_shape = metadata[key]
            expected = (
                (w_shape[0] + FP8_UE8M0_BLOCK_SIZE - 1) // FP8_UE8M0_BLOCK_SIZE,
                (w_shape[1] + FP8_UE8M0_BLOCK_SIZE - 1) // FP8_UE8M0_BLOCK_SIZE,
            )
            if w_dtype != "F8_E4M3" or s_dtype != "U8" or s_shape != expected:
                errors.append((base, metadata[w_key], metadata[key], expected))
            fp8_count += 1
    if errors:
        raise RuntimeError(f"Validation failed: {errors[:20]}")

    expected_mxfp4 = len(sparse_layers) * num_experts * 3
    if mxfp4_count != expected_mxfp4:
        raise RuntimeError(f"Expected {expected_mxfp4} MXFP4 modules, found {mxfp4_count}")
    if fp8_count == 0:
        raise RuntimeError("No FP8 UE8M0 modules found")
    if any(
        key.startswith("model.mtp_layers.") and key.endswith((".gate_up_proj", ".down_proj"))
        for key in keys
    ):
        raise RuntimeError("Fused BF16 MTP expert tensors remain in output")

    print(
        f"  Validation passed: {mxfp4_count} MXFP4, {fp8_count} FP8 modules",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    input_path,
    output_path,
    num_workers=8,
    use_gpu=True,
    do_validate=True,
):
    source_config = _load_json(os.path.join(input_path, "config.json"))
    if "quantization_config" in source_config:
        raise AssertionError(
            "Input checkpoint already has quantization_config; "
            "this script expects a BF16 checkpoint."
        )

    if use_gpu and torch.cuda.device_count() == 0:
        print("[warn] No CUDA device; falling back to CPU.", flush=True)
        use_gpu = False

    sparse_layers = sparse_main_layers(source_config)
    num_experts = int(source_config["n_routed_experts"])

    os.makedirs(output_path, exist_ok=True)
    manifest_dir = os.path.join(output_path, ".build_manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    shards = _collect_shards(input_path)
    tasks = [
        (shard, input_path, output_path, manifest_dir, tuple(sparse_layers), use_gpu)
        for shard in shards
    ]

    print(
        f"[HY4 MXFP4+FP8] shards={len(shards)}, "
        f"sparse_layers={sparse_layers[0]}..{sparse_layers[-1]}, "
        f"experts={num_experts}, workers={num_workers}",
        flush=True,
    )

    manifests = _run_tasks(tasks, num_workers)
    weight_map, total_size, mxfp4_total, fp8_total, ignored_modules = _merge_manifests(
        manifest_dir, manifests
    )

    _atomic_write_json(
        {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        },
        os.path.join(output_path, INDEX_NAME),
    )

    _copy_auxiliary_files(input_path, output_path)

    output_config = dict(source_config)
    output_config["quantization_config"] = _build_fp8_config(ignored_modules)
    output_config["mtp_quant_algo"] = "FP8"
    _atomic_write_json(output_config, os.path.join(output_path, "config.json"))
    _atomic_write_json(
        _build_mxfp4_sidecar(source_config, sparse_layers),
        os.path.join(output_path, "hf_quant_config.json.mxfp4"),
    )
    _atomic_write_json(
        _build_angelslim_sidecar(source_config, sparse_layers, mxfp4_total, fp8_total),
        os.path.join(output_path, "angelslim_config.json"),
    )

    if do_validate:
        print("[validate] Checking output checkpoint...", flush=True)
        validate_output(output_path, sparse_layers, num_experts)

    shutil.rmtree(manifest_dir, ignore_errors=True)

    print(
        f"[HY4 MXFP4+FP8] done: {output_path}\n"
        f"  MXFP4 expert modules: {mxfp4_total}\n"
        f"  FP8 UE8M0 modules: {fp8_total}\n"
        f"  total bytes: {total_size}",
        flush=True,
    )


def _load_yaml_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    quant = config["compression"]["quantization"]
    method = quant.get("quant_method", {})
    return {
        "num_workers": method.get("num_workers", 8),
        "use_gpu": method.get("use_gpu", True),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=str, help="Optional YAML config")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    kwargs = {}
    if args.config:
        kwargs = _load_yaml_config(args.config)

    kwargs.update(
        {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "do_validate": not args.skip_validate,
        }
    )
    if args.cpu:
        kwargs["use_gpu"] = False
    elif args.use_gpu:
        kwargs["use_gpu"] = True
    if args.num_workers:
        kwargs["num_workers"] = args.num_workers

    print(kwargs, flush=True)
    main(**kwargs)
