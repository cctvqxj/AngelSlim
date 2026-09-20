#!/usr/bin/env python3
"""Convert a HY4 MXFP4-RTN Stage-1 checkpoint to mixed FP8 UE8M0.

Input checkpoint requirements:

* main routed experts are already split per expert and stored as MXFP4
  ``weight`` / ``weight_scale`` pairs;
* shared experts, attention, dense MLP, MTP and other non-expert tensors are
  still BF16/FP32.

The Stage-1 expert weights must be produced by the data-free RTN converter.

Output policy:

* main routed experts remain unchanged MXFP4;
* main dense MLP, shared experts, attention projections, linear gate and
  indexer WK/WQ_B become FP8 E4M3 with 128x128 UE8M0 ``scale`` tensors;
* MTP attention/shared/routed-expert GEMM weights use the same FP8 UE8M0
  layout; fused MTP experts are split into per-expert keys;
* lm_head, embedding, router, norms, iHC tensors, learnable sink,
  ``eh_proj`` and indexer ``weights_proj`` remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

INDEX_NAME = "model.safetensors.index.json"
MX_BLOCK_SIZE = 32
MX_E8M0_BIAS = 127
FP8_UE8M0_BLOCK_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_MIN = float(torch.finfo(FP8_DTYPE).min)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON top level is not an object: {path}")
    return value


def atomic_write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def load_weight_map(model_path: str) -> tuple[dict[str, Any], dict[str, str]]:
    index_path = os.path.join(model_path, INDEX_NAME)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing {INDEX_NAME}: {model_path}")
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map: {index_path}")
    return index, weight_map


def sparse_main_layers(config: dict[str, Any]) -> list[int]:
    num_hidden_layers = int(config["num_hidden_layers"])
    layer_types = config.get("mlp_layer_types")
    if isinstance(layer_types, list) and len(layer_types) >= num_hidden_layers:
        layers = [
            index
            for index, layer_type in enumerate(layer_types[:num_hidden_layers])
            if str(layer_type).lower() in {"sparse", "moe"}
        ]
    else:
        layers = list(range(int(config.get("first_k_dense_replace", 1)), num_hidden_layers))
    if not layers:
        raise ValueError("No sparse/MoE main-model layers were found in config.json")
    return layers


def fp8_ue8m0_quantize(
    weight: torch.Tensor,
    block_size: int = FP8_UE8M0_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to FP8 E4M3 with 128x128 UE8M0 scales."""
    if block_size <= 0:
        raise ValueError(f"FP8 UE8M0 block_size must be positive, got {block_size}.")
    if weight.ndim != 2:
        raise ValueError(f"FP8 UE8M0 expects a 2-D weight, got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError(f"FP8 UE8M0 expects a floating weight, got {weight.dtype}")

    rows, columns = weight.shape
    row_blocks = (rows + block_size - 1) // block_size
    column_blocks = (columns + block_size - 1) // block_size
    padded_rows = row_blocks * block_size
    padded_columns = column_blocks * block_size
    source = weight.float()
    if padded_rows != rows or padded_columns != columns:
        source = torch.nn.functional.pad(
            source,
            (0, padded_columns - columns, 0, padded_rows - rows),
        )

    blocks = source.reshape(row_blocks, block_size, column_blocks, block_size)
    block_amax = blocks.abs().amax(dim=(1, 3))
    safe_amax = torch.where(block_amax > 0, block_amax, torch.ones_like(block_amax))
    exponent = torch.ceil(torch.log2(safe_amax / FP8_MAX))
    encoded_scale = (exponent + MX_E8M0_BIAS).clamp_(0, 255).to(torch.uint8)
    encoded_scale = torch.where(
        block_amax > 0,
        encoded_scale,
        torch.full_like(encoded_scale, MX_E8M0_BIAS),
    )
    decoded_scale = torch.pow(
        2.0,
        encoded_scale.to(torch.int32).float() - MX_E8M0_BIAS,
    )
    quantized = (
        (blocks / decoded_scale[:, None, :, None])
        .clamp_(FP8_MIN, FP8_MAX)
        .reshape(padded_rows, padded_columns)[:rows, :columns]
        .to(FP8_DTYPE)
        .contiguous()
    )
    return quantized, encoded_scale.contiguous()

MAIN_MXFP4_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale)$"
)
MTP_FUSED_EXPERT_RE = re.compile(
    r"^model\.mtp_layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<projection>gate_up_proj|down_proj)$"
)
MAIN_FP8_WEIGHT_RES = (
    re.compile(
        r"^model\.layers\.\d+\.self_attn\."
        r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|linear_gate)"
        r"\.weight$"
    ),
    re.compile(r"^model\.layers\.\d+\.self_attn\.indexer\.(?:wk|wq_b)\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"),
    re.compile(
        r"^model\.layers\.\d+\.mlp\.shared_experts\." r"(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
)
MTP_FP8_WEIGHT_RES = (
    re.compile(
        r"^model\.mtp_layers\.\d+\.self_attn\."
        r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|linear_gate)"
        r"\.weight$"
    ),
    re.compile(r"^model\.mtp_layers\.\d+\.self_attn\.indexer\.(?:wk|wq_b)\.weight$"),
    re.compile(r"^model\.mtp_layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"),
    re.compile(
        r"^model\.mtp_layers\.\d+\.mlp\.shared_experts\."
        r"(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
)


def prepare_output_directory(output_path: str, input_path: str) -> None:
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Input checkpoint does not exist: {source}")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must be outside the input checkpoint")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def is_main_mxfp4_tensor(name: str, sparse_layers: set[int]) -> bool:
    match = MAIN_MXFP4_RE.fullmatch(name)
    return bool(match and int(match.group("layer")) in sparse_layers)


def is_fp8_weight(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in (*MAIN_FP8_WEIGHT_RES, *MTP_FP8_WEIGHT_RES))


def _worker_device(use_gpu: bool) -> str:
    if not use_gpu or not torch.cuda.is_available():
        return "cpu"
    identity = mp.current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_idx = worker_idx % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    return f"cuda:{device_idx}"


def _store_fp8(
    state: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    output_name: str,
    key: str,
    weight: torch.Tensor,
    device: str,
) -> None:
    source = weight.to(device)
    quantized, scale = fp8_ue8m0_quantize(source)
    state[key] = quantized.cpu()
    scale_key = key[: -len(".weight")] + ".scale"
    state[scale_key] = scale.cpu()
    weight_map[key] = output_name
    weight_map[scale_key] = output_name
    del source, quantized, scale


def _store_mtp_experts(
    state: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    output_name: str,
    key: str,
    weight: torch.Tensor,
    device: str,
) -> int:
    match = MTP_FUSED_EXPERT_RE.fullmatch(key)
    if match is None:
        return 0
    prefix = key.rsplit(".experts.", 1)[0] + ".experts"
    projection = match.group("projection")
    count = 0
    for expert_idx in range(weight.shape[0]):
        if projection == "gate_up_proj":
            gate, up = weight[expert_idx].chunk(2, dim=0)
            projections = (("gate_proj", gate), ("up_proj", up))
        else:
            projections = (("down_proj", weight[expert_idx]),)
        for projection_name, expert_weight in projections:
            target = f"{prefix}.{expert_idx}.{projection_name}.weight"
            _store_fp8(
                state,
                weight_map,
                output_name,
                target,
                expert_weight,
                device,
            )
            count += 1
    return count


def _passthrough_module_name(key: str) -> str:
    for suffix in (".weight", ".bias"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def process_shard(task: tuple[Any, ...]) -> str:
    (
        shard_name,
        input_path,
        output_path,
        manifest_dir,
        sparse_layer_values,
        use_gpu,
    ) = task
    sparse_layers = set(sparse_layer_values)
    output_name = shard_name
    output_file = os.path.join(output_path, output_name)
    device = _worker_device(use_gpu)
    state: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    ignored_modules: set[str] = set()
    fp8_modules = 0
    mtp_expert_modules = 0
    mxfp4_tensors = 0

    with safe_open(
        os.path.join(input_path, shard_name),
        framework="pt",
        device="cpu",
    ) as reader:
        for key in reader.keys():
            tensor = reader.get_tensor(key)
            if is_main_mxfp4_tensor(key, sparse_layers):
                state[key] = tensor
                weight_map[key] = output_name
                mxfp4_tensors += 1
                continue

            if MTP_FUSED_EXPERT_RE.fullmatch(key):
                count = _store_mtp_experts(
                    state,
                    weight_map,
                    output_name,
                    key,
                    tensor,
                    device,
                )
                fp8_modules += count
                mtp_expert_modules += count
                del tensor
                continue

            if is_fp8_weight(key):
                _store_fp8(
                    state,
                    weight_map,
                    output_name,
                    key,
                    tensor,
                    device,
                )
                fp8_modules += 1
                del tensor
                continue

            state[key] = tensor
            weight_map[key] = output_name
            ignored_modules.add(_passthrough_module_name(key))

    save_file(state, output_file, metadata={"format": "pt"})
    manifest = {
        "weight_map": weight_map,
        "total_size": sum(tensor_nbytes(tensor) for tensor in state.values()),
        "fp8_modules": fp8_modules,
        "mtp_expert_modules": mtp_expert_modules,
        "mxfp4_tensors": mxfp4_tensors,
        "ignored_modules": sorted(ignored_modules),
    }
    manifest_name = shard_name + ".json"
    atomic_write_json(manifest, os.path.join(manifest_dir, manifest_name))
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest_name


def run_tasks(tasks: list[tuple[Any, ...]], workers: int) -> list[str]:
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        return [process_shard(task) for task in tqdm(tasks, desc="FP8 UE8M0")]
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        return list(
            tqdm(
                pool.imap_unordered(process_shard, tasks),
                total=len(tasks),
                desc="FP8 UE8M0",
            )
        )


def merge_manifests(manifest_dir: str, names: list[str]):
    weight_map: dict[str, str] = {}
    total_size = 0
    counters = CounterLike()
    ignored_modules: set[str] = set()
    for name in names:
        manifest = load_json(os.path.join(manifest_dir, name))
        overlap = set(weight_map).intersection(manifest["weight_map"])
        if overlap:
            raise RuntimeError(f"Duplicate output tensors: {sorted(overlap)[:20]}")
        weight_map.update(manifest["weight_map"])
        total_size += int(manifest["total_size"])
        for counter in ("fp8_modules", "mtp_expert_modules", "mxfp4_tensors"):
            counters[counter] += int(manifest.get(counter, 0))
        ignored_modules.update(manifest["ignored_modules"])
    return weight_map, total_size, counters, sorted(ignored_modules)


class CounterLike(dict):
    def __missing__(self, key):
        return 0


def build_mxfp4_sidecar(
    source_config: dict[str, Any],
    sparse_layers: list[int],
) -> dict[str, Any]:
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
        "producer": {
            "name": "AngelSlim",
            "tool": Path(__file__).name,
        },
        "quantization": {
            "quant_algo": "MXFP4",
            "kv_cache_quant_algo": None,
            "group_size": MX_BLOCK_SIZE,
            "exclude_modules": exclude,
            "num_hidden_layers": num_hidden_layers,
        },
    }


def build_fp8_config(ignored_modules: list[str]) -> dict[str, Any]:
    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [
            FP8_UE8M0_BLOCK_SIZE,
            FP8_UE8M0_BLOCK_SIZE,
        ],
        "modules_to_not_convert": ignored_modules,
        "scale_fmt": "ue8m0",
    }


def validate_rtn_stage1(input_path: str) -> None:
    """Reject checkpoints that do not explicitly declare MXFP4 RTN."""
    config_path = os.path.join(input_path, "config.json")
    config = load_json(config_path)
    rtn_config = config.get("angelslim_mxfp4_config")
    if not isinstance(rtn_config, dict) or rtn_config.get("algorithm") != "rtn":
        raise ValueError("Stage-1 checkpoint is not marked as data-free MXFP4 RTN")


def build_angelslim_sidecar(input_path: str) -> dict[str, Any]:
    """Preserve Stage-1 expert policy while updating the final MTP format."""
    source = os.path.join(input_path, "angelslim_config.json")
    source_sidecar = load_json(source) if os.path.isfile(source) else {}
    source_rtn_sidecar = source_sidecar.get("rtn_config")
    sidecar = (
        {"rtn_config": source_rtn_sidecar}
        if isinstance(source_rtn_sidecar, dict)
        else {}
    )

    source_config = load_json(os.path.join(input_path, "config.json"))
    source_rtn_config = source_config.get("angelslim_mxfp4_config", {})
    rtn_config = sidecar.get("rtn_config")
    if not isinstance(rtn_config, dict):
        rtn_config = {}
    else:
        rtn_config = dict(rtn_config)
    if isinstance(source_rtn_config, dict):
        for field in (
            "weight_format",
            "scale_format",
            "group_size",
            "include_patterns",
            "exclude_patterns",
            "quantized_module_count",
        ):
            if field in source_rtn_config:
                rtn_config[field] = source_rtn_config[field]
    rtn_config.update(
        {
            "algorithm": "rtn",
            "mtp_quant_algo": "FP8",
            "mtp_quant_method": "rtn",
        }
    )
    sidecar["rtn_config"] = rtn_config
    return sidecar


def copy_auxiliary_files(input_path: str, output_path: str) -> None:
    generated = {
        INDEX_NAME,
        "config.json",
        "angelslim_config.json",
        "hf_quant_config.json",
    }
    for name in os.listdir(input_path):
        source = os.path.join(input_path, name)
        if not os.path.isfile(source):
            continue
        if (
            name in generated
            or name.startswith("hf_quant_config.json.")
            or name == "expert_quantization_report.json"
            or name.endswith((".safetensors", ".bin"))
        ):
            continue
        if name.endswith((".json", ".jinja", ".py", ".md", ".txt", ".model")):
            shutil.copy2(source, os.path.join(output_path, name))


def validate_output(
    output_path: str,
    sparse_layers: list[int],
    num_experts: int,
) -> None:
    index = load_json(os.path.join(output_path, INDEX_NAME))
    weight_map = index["weight_map"]
    keys = set(weight_map)
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    metadata: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard, shard_keys in by_shard.items():
        path = os.path.join(output_path, shard)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing output shard: {shard}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(shard_keys):
                raise RuntimeError(f"Stored/indexed key mismatch in {shard}")
            for key in shard_keys:
                tensor = reader.get_slice(key)
                metadata[key] = (
                    str(tensor.get_dtype()),
                    tuple(tensor.get_shape()),
                )

    missing = []
    for layer in sparse_layers:
        for expert in range(num_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                for suffix in (".weight", ".weight_scale"):
                    if base + suffix not in keys:
                        missing.append(base + suffix)
    if missing:
        raise RuntimeError(f"Missing MXFP4 tensors: {missing[:20]}")

    mxfp4_modules = 0
    fp8_modules = 0
    errors = []
    for key in keys:
        if key.endswith(".weight_scale"):
            base = key[: -len(".weight_scale")]
            match = MAIN_MXFP4_RE.fullmatch(key)
            if match is None:
                errors.append((key, "unexpected weight_scale"))
                continue
            weight_key = base + ".weight"
            weight_dtype, weight_shape = metadata[weight_key]
            scale_dtype, scale_shape = metadata[key]
            expected = (weight_shape[0], weight_shape[1] * 2 // MX_BLOCK_SIZE)
            if weight_dtype != "U8" or scale_dtype != "U8" or scale_shape != expected:
                errors.append((base, metadata[weight_key], metadata[key], expected))
            mxfp4_modules += 1
        elif key.endswith(".scale"):
            base = key[: -len(".scale")]
            weight_key = base + ".weight"
            if weight_key not in keys:
                errors.append((key, "missing weight"))
                continue
            weight_dtype, weight_shape = metadata[weight_key]
            scale_dtype, scale_shape = metadata[key]
            expected = (
                (weight_shape[0] + FP8_UE8M0_BLOCK_SIZE - 1) // FP8_UE8M0_BLOCK_SIZE,
                (weight_shape[1] + FP8_UE8M0_BLOCK_SIZE - 1) // FP8_UE8M0_BLOCK_SIZE,
            )
            if weight_dtype != "F8_E4M3" or scale_dtype != "U8" or scale_shape != expected:
                errors.append((base, metadata[weight_key], metadata[key], expected))
            fp8_modules += 1
    if errors:
        raise RuntimeError(f"Quantized tensor validation failed: {errors[:20]}")

    expected_mxfp4 = len(sparse_layers) * num_experts * 3
    if mxfp4_modules != expected_mxfp4:
        raise RuntimeError(f"Expected {expected_mxfp4} MXFP4 modules, found {mxfp4_modules}")
    if fp8_modules == 0:
        raise RuntimeError("No FP8 UE8M0 modules found")
    if any(
        key.startswith("model.mtp_layers.") and key.endswith((".gate_up_proj", ".down_proj"))
        for key in keys
    ):
        raise RuntimeError("Fused BF16 MTP expert tensors remain in output")

    disk_shards = {os.path.basename(path) for path in Path(output_path).glob("*.safetensors")}
    if disk_shards != set(by_shard):
        raise RuntimeError(
            f"Disk/index shard mismatch: extra={sorted(disk_shards - set(by_shard))[:10]}, "
            f"missing={sorted(set(by_shard) - disk_shards)[:10]}"
        )


def main(
    input_path: str,
    output_path: str,
    num_workers: int = 8,
    use_gpu: bool = True,
    validate: bool = True,
) -> None:
    validate_rtn_stage1(input_path)
    prepare_output_directory(output_path, input_path)
    manifest_dir = os.path.join(output_path, ".build_manifests")
    os.makedirs(manifest_dir)

    source_config = load_json(os.path.join(input_path, "config.json"))
    sparse_layers = sparse_main_layers(source_config)
    num_experts = int(source_config["n_routed_experts"])
    _, source_weight_map = load_weight_map(input_path)
    shards = sorted(set(source_weight_map.values()))
    tasks = [
        (
            shard,
            input_path,
            output_path,
            manifest_dir,
            tuple(sparse_layers),
            use_gpu,
        )
        for shard in shards
    ]

    print(
        f"[HY4 MXFP4+FP8] source_shards={len(shards)}, "
        f"sparse_layers={sparse_layers[0]}..{sparse_layers[-1]}, "
        f"experts={num_experts}, expert_method=rtn"
    )
    manifests = run_tasks(tasks, num_workers)
    weight_map, total_size, counters, ignored_modules = merge_manifests(
        manifest_dir,
        manifests,
    )
    atomic_write_json(
        {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        },
        os.path.join(output_path, INDEX_NAME),
    )

    output_config = dict(source_config)
    output_config["quantization_config"] = build_fp8_config(ignored_modules)
    output_config["mtp_quant_algo"] = "FP8"
    output_config["angelslim_mixed_mxfp4_fp8_config"] = {
        "algorithm": "preserve_mxfp4_rtn_quantize_dense_fp8_ue8m0",
        "expert_quant_method": "rtn",
        "mxfp4_main_expert_layers": sparse_layers,
        "mxfp4_main_expert_module_count": len(sparse_layers) * num_experts * 3,
        "fp8_ue8m0_module_count": counters["fp8_modules"],
        "fp8_ue8m0_mtp_expert_module_count": counters["mtp_expert_modules"],
        "fp8_block_size": [
            FP8_UE8M0_BLOCK_SIZE,
            FP8_UE8M0_BLOCK_SIZE,
        ],
        "fp8_scale_format": "ue8m0",
        "fp8_scale_tensor_name": "scale",
    }
    atomic_write_json(output_config, os.path.join(output_path, "config.json"))
    atomic_write_json(
        build_mxfp4_sidecar(source_config, sparse_layers),
        os.path.join(output_path, "hf_quant_config.json.mxfp4"),
    )
    atomic_write_json(
        build_angelslim_sidecar(input_path),
        os.path.join(output_path, "angelslim_config.json"),
    )
    copy_auxiliary_files(input_path, output_path)

    if validate:
        validate_output(output_path, sparse_layers, num_experts)
    shutil.rmtree(manifest_dir)
    print(
        f"[HY4 MXFP4+FP8] done: {output_path}\n"
        f"  MXFP4 expert modules: {len(sparse_layers) * num_experts * 3}\n"
        f"  FP8 UE8M0 modules: {counters['fp8_modules']}\n"
        f"  MTP FP8 expert modules: {counters['mtp_expert_modules']}\n"
        f"  logical tensor bytes: {total_size}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        num_workers=args.num_workers,
        use_gpu=not args.cpu,
        validate=not args.skip_validate,
    )
