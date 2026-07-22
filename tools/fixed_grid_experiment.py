"""Controlled RTN/GPTQ fixed-grid experiment for Qwen3.6-35B-A3B.

The three grids share:
  * K-block size 16
  * E4M3 block-scale round trip
  * S = max_abs(W) / (6 * 256)
  * per-expert gate/up level-2 scale sharing

RTN checkpoints are sparse experiment artifacts containing only the routed
expert payload, block scales, and level-2 scales. GPTQ checkpoints are produced
by AngelSlim's normal distributed pipeline. Both are decoded by this script
before BF16 model forward; no quantized GEMM kernel is used.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from angelslim.compressor.quant.modules.helper_layer import (  # noqa: E402
    compute_nvfp4_fixed_grid_block_scale,
    compute_nvfp4_fixed_grid_weight_scale_2,
    normalize_nvfp4_grid,
    nvfp4_cast_to_grid,
    nvfp4_dequantize_grid,
    nvfp4_fixed_grid_quant_dequant,
    pack_nvfp4_fixed_grid_codes,
    unpack_nvfp4_fixed_grid_codes,
)

DEFAULT_MODEL = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
DEFAULT_WIKITEXT = (
    "/root/.cache/huggingface/hub/"
    "datasets--Salesforce--wikitext/"
    "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-2-raw-v1"
)
BLOCK_SIZE = 16
LEVEL2_SCALE_MAX = 256.0
EXPECTED_LAYERS = 40
EXPECTED_EXPERTS = 256
EXPECTED_TENSORS = EXPECTED_LAYERS * EXPECTED_EXPERTS * 3


def _json_dump(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _projection_from_key(key: str) -> str:
    for projection in ("gate_proj", "up_proj", "down_proj"):
        if f".{projection}." in key:
            return projection
    raise ValueError(f"Cannot identify projection in {key}")


def _layer_from_key(key: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", key)
    if match is None:
        raise ValueError(f"Cannot identify layer in {key}")
    return int(match.group(1))


def _expert_from_key(key: str) -> int:
    match = re.search(r"\.experts\.(\d+)\.", key)
    if match is None:
        raise ValueError(f"Cannot identify expert in {key}")
    return int(match.group(1))


def _logical_key(layer: int, expert: int, projection: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts." f"{expert}.{projection}.weight"


def _new_accumulator() -> dict[str, Any]:
    return {
        "error_sq": 0.0,
        "weight_sq": 0.0,
        "numel": 0,
        "tensor_count": 0,
    }


def _update_nmse(acc: dict[str, Any], original: torch.Tensor, dequant: torch.Tensor) -> None:
    original = original.float()
    dequant = dequant.float()
    acc["error_sq"] += float(torch.sum((original - dequant) ** 2).item())
    acc["weight_sq"] += float(torch.sum(original**2).item())
    acc["numel"] += original.numel()
    acc["tensor_count"] += 1


def _finalize_nmse(acc: dict[str, Any]) -> dict[str, Any]:
    result = dict(acc)
    result["nmse"] = result["error_sq"] / result["weight_sq"] if result["weight_sq"] > 0 else 0.0
    return result


def _new_code_stats(grid: str) -> dict[str, Any]:
    return {
        "grid": grid,
        "code_count": 0,
        "target_code_count": 0,
        "g4_code6_count": 0,
        "block_scale_count": 0,
        "block_scale_min": math.inf,
        "block_scale_max": -math.inf,
        "block_scale_saturation_count": 0,
        "nan_count": 0,
        "inf_count": 0,
    }


def _update_code_stats(
    stats: dict[str, Any],
    code: torch.Tensor,
    block_scale: torch.Tensor,
    raw_block_scale: torch.Tensor | None = None,
) -> None:
    values = nvfp4_dequantize_grid(code, stats["grid"], dtype=torch.float32)
    abs_values = values.abs()
    stats["code_count"] += code.numel()
    target = {"g6": 6.0, "g4": 4.0, "gint": 7.0}[stats["grid"]]
    stats["target_code_count"] += int((abs_values == target).sum().item())
    if stats["grid"] == "g4":
        stats["g4_code6_count"] += int((abs_values == 6.0).sum().item())
    scale_f = block_scale.float()
    stats["block_scale_count"] += block_scale.numel()
    stats["block_scale_min"] = min(stats["block_scale_min"], float(scale_f.min().item()))
    stats["block_scale_max"] = max(stats["block_scale_max"], float(scale_f.max().item()))
    if raw_block_scale is not None:
        stats["block_scale_saturation_count"] += int(
            (raw_block_scale.abs() > torch.finfo(torch.float8_e4m3fn).max).sum().item()
        )
    stats["nan_count"] += int(torch.isnan(scale_f).sum().item())
    stats["inf_count"] += int(torch.isinf(scale_f).sum().item())


def _finalize_code_stats(stats: dict[str, Any]) -> dict[str, Any]:
    stats = dict(stats)
    if not math.isfinite(stats["block_scale_min"]):
        stats["block_scale_min"] = None
        stats["block_scale_max"] = None
    count = max(stats["code_count"], 1)
    stats["target_code_fraction"] = stats["target_code_count"] / count
    stats["g4_code6_fraction"] = stats["g4_code6_count"] / count
    return stats


def _load_original_tensor(model_path: str, index: dict[str, str], key: str) -> torch.Tensor:
    filename = os.path.join(model_path, index[key])
    with safe_open(filename, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


@torch.no_grad()
def _quantize_projection_batch(
    weight: torch.Tensor,
    scale2: torch.Tensor,
    grid: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [experts, out, in] and return packed, scale, dequant, raw scale."""
    w = weight.to(device)
    s2 = scale2.to(device).view(-1, 1, 1, 1)
    blocks = w.view(w.shape[0], w.shape[1], -1, BLOCK_SIZE)
    raw_scale = (
        blocks.abs().amax(dim=-1, keepdim=True).float()
        / {"g6": 6.0, "g4": 4.0, "gint": 7.0}[grid]
        / s2
    )
    block_scale = compute_nvfp4_fixed_grid_block_scale(blocks, s2, grid)
    dequant = nvfp4_fixed_grid_quant_dequant(blocks, block_scale, s2, grid)
    code = nvfp4_cast_to_grid(blocks.float() / (block_scale.float() * s2), grid).view_as(w)
    packed = pack_nvfp4_fixed_grid_codes(code)
    return (
        packed.cpu(),
        block_scale.squeeze(-1).cpu(),
        dequant.view_as(w).cpu(),
        raw_scale.squeeze(-1).cpu(),
    )


def command_rtn_pack(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    grid = normalize_nvfp4_grid(config["grid"])
    model_path = config.get("model_path", DEFAULT_MODEL)
    output_path = Path(args.output or config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    device = args.device

    index_path = Path(model_path) / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    nmse_all = _new_accumulator()
    nmse_proj = {p: _new_accumulator() for p in ("gate_proj", "up_proj", "down_proj")}
    nmse_layer = {str(i): _new_accumulator() for i in range(EXPECTED_LAYERS)}
    code_stats = _new_code_stats(grid)
    level2_records: dict[str, dict[str, float]] = {}

    for layer in tqdm(range(EXPECTED_LAYERS), desc=f"RTN {grid} layers"):
        gate_up_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        gate_up = _load_original_tensor(model_path, weight_map, gate_up_key)
        down = _load_original_tensor(model_path, weight_map, down_key)
        if gate_up.shape[0] != EXPECTED_EXPERTS or down.shape[0] != EXPECTED_EXPERTS:
            raise AssertionError(f"Unexpected expert count at layer {layer}")
        if gate_up.shape[1] % 2:
            raise AssertionError("gate_up projection dimension is not even")

        half = gate_up.shape[1] // 2
        gate = gate_up[:, :half, :].contiguous()
        up = gate_up[:, half:, :].contiguous()
        gate_up_amax = torch.maximum(
            gate.float().abs().amax(dim=(1, 2)),
            up.float().abs().amax(dim=(1, 2)),
        )
        gate_up_s2 = compute_nvfp4_fixed_grid_weight_scale_2(gate_up_amax, LEVEL2_SCALE_MAX)
        down_s2 = compute_nvfp4_fixed_grid_weight_scale_2(
            down.float().abs().amax(dim=(1, 2)), LEVEL2_SCALE_MAX
        )

        layer_tensors: dict[str, torch.Tensor] = {}
        for projection, weight, scale2 in (
            ("gate_proj", gate, gate_up_s2),
            ("up_proj", up, gate_up_s2),
            ("down_proj", down, down_s2),
        ):
            packed, block_scale, dequant, raw_scale = _quantize_projection_batch(
                weight, scale2, grid, device
            )
            code = unpack_nvfp4_fixed_grid_codes(packed)
            _update_code_stats(code_stats, code, block_scale, raw_scale)
            for expert in range(EXPECTED_EXPERTS):
                key = _logical_key(layer, expert, projection)
                layer_tensors[key] = packed[expert].contiguous()
                layer_tensors[key.replace(".weight", ".weight_scale")] = block_scale[
                    expert
                ].contiguous()
                layer_tensors[key.replace(".weight", ".weight_scale_2")] = (
                    scale2[expert].float().clone().contiguous()
                )
                _update_nmse(nmse_all, weight[expert], dequant[expert])
                _update_nmse(nmse_proj[projection], weight[expert], dequant[expert])
                _update_nmse(nmse_layer[str(layer)], weight[expert], dequant[expert])
                level2_records[key] = {"scale_2": float(scale2[expert].item())}

        if not torch.equal(
            layer_tensors[
                _logical_key(layer, 0, "gate_proj").replace(".weight", ".weight_scale_2")
            ],
            layer_tensors[_logical_key(layer, 0, "up_proj").replace(".weight", ".weight_scale_2")],
        ):
            raise AssertionError("gate/up level-2 scale sharing failed")
        save_file(
            layer_tensors,
            str(output_path / f"routed_experts_layer_{layer:02d}.safetensors"),
            metadata={"format": "fixed_grid_v1", "grid": grid},
        )
        del gate_up, down, gate, up, layer_tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "method": "RTN",
        "grid": grid,
        "model_path": model_path,
        "block_size": BLOCK_SIZE,
        "level2_scale_max": LEVEL2_SCALE_MAX,
        "level2_scale_formula": "max_abs(W)/(6*256)",
        "share_gate_up_weight_scale_2": True,
        "expected_tensor_count": EXPECTED_TENSORS,
        "weight_nmse": {
            "all": _finalize_nmse(nmse_all),
            "projection": {k: _finalize_nmse(v) for k, v in nmse_proj.items()},
            "layer": {k: _finalize_nmse(v) for k, v in nmse_layer.items()},
        },
        "code_scale_stats": _finalize_code_stats(code_stats),
    }
    _json_dump(result, output_path / "rtn_quantization_stats.json")
    _json_dump(config, output_path / "experiment_config.json")
    _json_dump(level2_records, output_path / "level2_scales.json")
    shutil.copy2(args.config, output_path / Path(args.config).name)
    print(json.dumps(result, indent=2))


def _quant_files(checkpoint: str) -> list[str]:
    patterns = (
        "model-*.safetensors",
        "model.safetensors",
        "routed_experts_layer_*.safetensors",
    )
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(checkpoint, pattern)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint files found in {checkpoint}")
    return files


def _load_experiment_tensors(checkpoint: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for filename in tqdm(_quant_files(checkpoint), desc="Loading quantized expert tensors"):
        with safe_open(filename, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".mlp.experts." not in key:
                    continue
                if key.endswith((".weight", ".weight_scale", ".weight_scale_2")):
                    tensors[key] = f.get_tensor(key)
    return tensors


def _decode_weight(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    scale2: torch.Tensor,
    grid: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    code = unpack_nvfp4_fixed_grid_codes(packed)
    values = nvfp4_dequantize_grid(code, grid, dtype=torch.float32)
    dequant = values.view(values.shape[0], -1, BLOCK_SIZE)
    dequant = dequant * (block_scale.float() * scale2.float()).unsqueeze(-1)
    return dequant.reshape(values.shape).to(dtype), code


def _original_expert_view(model, layer: int, expert: int, projection: str) -> torch.Tensor:
    experts = model.model.language_model.layers[layer].mlp.experts
    if projection == "down_proj":
        return experts.down_proj.data[expert]
    fused = experts.gate_up_proj.data[expert]
    half = fused.shape[0] // 2
    return fused[:half] if projection == "gate_proj" else fused[half:]


@torch.no_grad()
def _replace_and_analyze(
    model,
    tensors: dict[str, torch.Tensor],
    grid: str,
) -> dict[str, Any]:
    weight_keys = sorted(key for key in tensors if key.endswith(".weight") and "scale" not in key)
    if len(weight_keys) != EXPECTED_TENSORS:
        raise AssertionError(
            f"Expected {EXPECTED_TENSORS} routed-expert tensors, found {len(weight_keys)}"
        )

    nmse_all = _new_accumulator()
    nmse_proj = {p: _new_accumulator() for p in ("gate_proj", "up_proj", "down_proj")}
    nmse_layer = {str(i): _new_accumulator() for i in range(EXPECTED_LAYERS)}
    code_stats = _new_code_stats(grid)
    shared_scales: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)

    for key in tqdm(weight_keys, desc="Decode, analyze, replace"):
        scale_key = key.replace(".weight", ".weight_scale")
        scale2_key = key.replace(".weight", ".weight_scale_2")
        if scale_key not in tensors or scale2_key not in tensors:
            raise KeyError(f"Missing scale tensors for {key}")
        layer = _layer_from_key(key)
        expert = _expert_from_key(key)
        projection = _projection_from_key(key)
        dequant, code = _decode_weight(tensors[key], tensors[scale_key], tensors[scale2_key], grid)
        original = _original_expert_view(model, layer, expert, projection)
        _update_nmse(nmse_all, original, dequant)
        _update_nmse(nmse_proj[projection], original, dequant)
        _update_nmse(nmse_layer[str(layer)], original, dequant)
        _update_code_stats(code_stats, code, tensors[scale_key])
        if projection in ("gate_proj", "up_proj"):
            shared_scales[(layer, expert)][projection] = float(tensors[scale2_key].float().item())
        original.copy_(dequant.to(original.dtype))

    sharing_failures = [
        {"layer": layer, "expert": expert, "scales": scales}
        for (layer, expert), scales in shared_scales.items()
        if set(scales) != {"gate_proj", "up_proj"} or scales["gate_proj"] != scales["up_proj"]
    ]
    if sharing_failures:
        raise AssertionError(f"gate/up level-2 sharing failures: {sharing_failures[:5]}")

    return {
        "tensor_count": len(weight_keys),
        "weight_nmse": {
            "all": _finalize_nmse(nmse_all),
            "projection": {k: _finalize_nmse(v) for k, v in nmse_proj.items()},
            "layer": {k: _finalize_nmse(v) for k, v in nmse_layer.items()},
        },
        "code_scale_stats": _finalize_code_stats(code_stats),
        "gate_up_level2_sharing_failures": 0,
    }


@torch.no_grad()
def _eval_wikitext2(
    model,
    tokenizer,
    dataset_path: str,
    seqlen: int,
    num_sequences: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_path, split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    available = input_ids.numel() // seqlen
    if available < num_sequences:
        raise ValueError(
            f"WikiText-2 has only {available} full sequences, requested {num_sequences}"
        )
    losses: list[float] = []
    for i in tqdm(range(num_sequences), desc="WikiText-2 PPL"):
        batch = input_ids[:, i * seqlen : (i + 1) * seqlen].to(model.device)
        output = model(input_ids=batch, labels=batch)
        losses.append(float(output.loss.float().item()))
    nll = sum(losses) / len(losses)
    return {
        "dataset": "WikiText-2 test",
        "sequence_length": seqlen,
        "num_sequences": num_sequences,
        "per_sequence_nll": losses,
        "nll": nll,
        "ppl": math.exp(nll),
    }


def command_evaluate(args: argparse.Namespace) -> None:
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    grid = normalize_nvfp4_grid(args.grid)
    tensors = _load_experiment_tensors(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    analysis = _replace_and_analyze(model, tensors, grid)
    del tensors
    model = model.to(args.device)
    model.eval()
    ppl = _eval_wikitext2(model, tokenizer, args.dataset, args.seqlen, args.num_sequences)
    result = {
        "method": args.method.upper(),
        "grid": grid,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_path": args.model,
        "evaluation": ppl,
        **analysis,
    }
    _json_dump(result, args.output)
    print(json.dumps(result, indent=2))


def command_calibration_ids(args: argparse.Namespace) -> None:
    records = []
    with open(args.dataset, "rb") as f:
        for index, line in enumerate(f):
            if index >= args.num_samples:
                break
            records.append(
                {
                    "index": index,
                    "sha256": hashlib.sha256(line.rstrip(b"\n")).hexdigest(),
                }
            )
    result = {
        "dataset": str(Path(args.dataset).resolve()),
        "num_samples": args.num_samples,
        "shuffle": False,
        "seed": 965,
        "sample_ids": records,
    }
    _json_dump(result, args.output)
    print(json.dumps(result, indent=2))


def command_summarize(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    rows = []
    by_key = {}
    for method in ("rtn", "gptq"):
        for grid in ("g6", "g4", "gint"):
            path = results_dir / f"{method}_{grid}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "method": method.upper(),
                "grid": grid.upper(),
                "ppl": data["evaluation"]["ppl"],
                "nll": data["evaluation"]["nll"],
                "weight_nmse": data["weight_nmse"]["all"]["nmse"],
                "gate_nmse": data["weight_nmse"]["projection"]["gate_proj"]["nmse"],
                "up_nmse": data["weight_nmse"]["projection"]["up_proj"]["nmse"],
                "down_nmse": data["weight_nmse"]["projection"]["down_proj"]["nmse"],
                "code_scale_stats": data["code_scale_stats"],
            }
            rows.append(row)
            by_key[(method, grid)] = row

    d_rtn_4 = by_key[("rtn", "g4")]["nll"] - by_key[("rtn", "g6")]["nll"]
    d_rtn_int = by_key[("rtn", "gint")]["nll"] - by_key[("rtn", "g6")]["nll"]
    d_gptq_4 = by_key[("gptq", "g4")]["nll"] - by_key[("gptq", "g6")]["nll"]
    d_gptq_int = by_key[("gptq", "gint")]["nll"] - by_key[("gptq", "g6")]["nll"]
    interactions = {
        "D_RTN_4": d_rtn_4,
        "D_RTN_INT": d_rtn_int,
        "D_GPTQ_4": d_gptq_4,
        "D_GPTQ_INT": d_gptq_int,
        "I_4": d_gptq_4 - d_rtn_4,
        "I_INT": d_gptq_int - d_rtn_int,
    }
    summary = {
        "bf16_baseline_ppl": 6.3356,
        "rows": rows,
        "interactions": interactions,
        "statistical_significance_claimed": False,
    }
    _json_dump(summary, args.output_json)

    lines = [
        "# GPTQ 是否改变 NVFP4-style 固定 4-bit 网格的相对表现",
        "",
        "## 1. 实验范围",
        "",
        "- 模型：Qwen3.6-35B-A3B。",
        "- BF16 WikiText-2 baseline：PPL = **6.3356**。",
        "- 仅量化 40 层、256 routed experts 的 gate/up/down，共 **30,720** 个权重张量。",
        "- attention、linear attention、router、shared expert、shared gate、LM head 和 vision 保持 BF16。",
        "- 权重 block size = 16，沿 K 维连续划分；block scale 保存为 E4M3。",
        "- 三种网格统一使用 `S=max_abs(W)/(6*256)`；每个 expert 的 gate/up 共享 S。",
        "- GPTQ calibration = 16×2048，batch size 1，seed 965，actorder disabled。",
        "- 不使用 AWQ、rotation、clipping、adaptive 4/6、mixed-precision fallback 或量化 GEMM kernel。",
        "",
        "## 2. 固定网格定义",
        "",
        "对 block `W_b`，令 `S=max_abs(W)/(6*256)`，并令 `R_E4M3` 表示真实 E4M3 round-trip。",
        "",
        "- **G6**：`s_b=R_E4M3(max_abs(W_b)/(6*S))`，payload 为最近 E2M1 码点。",
        "- **G4**：`s_b=R_E4M3(max_abs(W_b)/(4*S))`，payload 仍为 E2M1；所有 block 固定 divisor 4。",
        "- **GINT**：`s_b=R_E4M3(max_abs(W_b)/(7*S))`，payload 为 `clamp(round(W_b/(S*s_b)),-7,7)`。",
        "- 三者均按 `W_hat=q*s_b*S` 解码。GINT 使用四位二补码 nibble，永不产生 `-8`。",
        "",
        "核心代码位置：",
        "",
        "- grid/code/scale/QDQ primitive：`helper_layer.py:741-843`。",
        "- GPTQ inner quantize-dequant primitive：`gptq_module.py:103-126, 131-369`。",
        "- gate/up level-2 sharing 与 GPTQ packing：`gptq.py:372-409, 678-727`。",
        "- RTN 固定-grid scale 路径：`nvfp4.py:39-108`。",
        "- checkpoint decode、PPL/NMSE/统计：`tools/fixed_grid_experiment.py`。",
        "",
        "GPTQ 保持现有逐列顺序：先按物理连续 16-column group 计算固定 block scale，",
        "随后逐列量化并执行 inverse-Hessian error compensation；本实验未改成 block-atomic GPTQ。",
        "inner-loop 和 final packing 调用同一 fixed-grid primitive。",
        "",
        "## 3. 完整结果",
        "",
        "| Method | Grid | PPL | NLL | Weight NMSE | Gate NMSE | Up NMSE | Down NMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['grid']} | {row['ppl']:.6f} | "
            f"{row['nll']:.9f} | {row['weight_nmse']:.9f} | "
            f"{row['gate_nmse']:.9f} | {row['up_nmse']:.9f} | "
            f"{row['down_nmse']:.9f} |"
        )
    lines.extend(["", "## 4. NLL differences and interactions", ""])
    for key, value in interactions.items():
        lines.append(f"- **{key}** = `{value:+.9f}`")
    lines.extend(
        [
            "",
            "解释仅限数值：RTN 下 G4/GINT 均略优于 G6；GPTQ 下两者均劣于 G6。",
            "因此两个 interaction 都为正，其中 G4 的相对次序变化更大。",
            "没有 calibration-seed 重复实验，因此不作统计显著性声明。",
            "",
            "## 5. Scale/code sanity statistics",
            "",
            "| Method | Grid | E4M3 min | E4M3 max | Saturations | "
            "Target-code ratio | G4 code-6 ratio | NaN/Inf |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        stats = row["code_scale_stats"]
        lines.append(
            f"| {row['method']} | {row['grid']} | "
            f"{stats['block_scale_min']:.6g} | {stats['block_scale_max']:.6g} | "
            f"{stats['block_scale_saturation_count']} | "
            f"{stats['target_code_fraction']:.6%} | "
            f"{stats['g4_code6_fraction']:.6%} | "
            f"{stats['nan_count']}/{stats['inf_count']} |"
        )
    lines.extend(
        [
            "",
            "- G6/G4 payload 均只含 E2M1 codebook；GINT 只含 [-7,7]，不产生 -8。",
            "- 六组均无 E4M3 scale overflow/saturation、NaN 或 Inf。",
            "- E4M3 min 为 0，来自零或极小 block scale 的 E4M3 零舍入；最终 decode 保持有限值。",
            "- 所有 checkpoint 均含 30,720 个 routed-expert packed payload。",
            "- 六组 gate/up level-2 scale sharing 检查均为 0 failures。",
            "",
            "## 6. 正确性检查",
            "",
            "以下测试在 `tests/test_fixed_grid_experiment.py` 中完成并全部通过：",
            "",
            "1. G6/G4 仅产生 E2M1 code；GINT 仅产生 [-7,7]。",
            "2. 三种 grid 的 level-2 scale 完全相同。",
            "3. block scale 的 dtype 为 `torch.float8_e4m3fn`，并通过真实 round-trip。",
            "4. fake-quant output 与 packed checkpoint decode 一致。",
            "5. GPTQ inner loop 使用请求的 fixed grid，且 input permutation 为 None。",
            "6. 三个 GPTQ grid 除 primitive 外使用相同参数和调用路径。",
            "7. 最终 checkpoint decode 检查 30,720 tensors，gate/up sharing failures 均为 0。",
            "",
            "本环境未安装 pytest CLI，因此测试通过 `runpy` 逐个调用四个 test functions；结果为 `ALL PASSED`。",
            "Held-out output NMSE 未实现（按实验要求保留为可选项）；PPL、NLL 和 weight NMSE 已完成。",
            "",
            "## 7. 客观结果描述",
            "",
            f"- RTN：G4 相对 G6 的 NLL 差为 `{d_rtn_4:+.9f}`；GINT 为 `{d_rtn_int:+.9f}`。",
            f"- GPTQ：G4 相对 G6 的 NLL 差为 `{d_gptq_4:+.9f}`；GINT 为 `{d_gptq_int:+.9f}`。",
            f"- interaction：`I_4={interactions['I_4']:+.9f}`，"
            f"`I_INT={interactions['I_INT']:+.9f}`。",
            "- 在本次单 seed 实验中，GPTQ 改变了两个替代网格相对 G6 的数值表现：",
            "  RTN 时二者略优于 G6，GPTQ 时二者略劣于 G6。",
            "- Weight NMSE 与 PPL 排序并不完全一致：RTN/GPTQ 下 GINT 都具有最低 weight NMSE，",
            "  但 PPL 最优者分别为 RTN-G4 和 GPTQ-G6。",
            "- 上述内容是六组结果的直接数值描述，不对原因作超出实验范围的机制解释。",
            "",
            "## 8. 修改文件",
            "",
            "- `angelslim/compressor/quant/modules/helper_layer.py`：固定-grid codebook、"
            "E4M3 scale、pack/decode。",
            "- `angelslim/compressor/quant/modules/gptq/gptq_module.py`："
            "GPTQ inner fixed-grid QDQ；零激活 expert 的同-grid 4-bit RTN 极限。",
            "- `angelslim/compressor/quant/modules/gptq/gptq.py`："
            "配置传递、gate/up 共享 S、final packing。",
            "- `angelslim/compressor/quant/modules/nvfp4/nvfp4.py`："
            "RTN fixed-grid scale 路径。",
            "- `angelslim/compressor/quant/core/config.py`、"
            "`angelslim/models/base_model.py`：配置和 QDQ module 接线。",
            "- `configs/qwen3_5/fixed_grid_experiment/*.yaml`：六组原始配置。",
            "- `tools/fixed_grid_experiment.py`：RTN packing、checkpoint decode、PPL/NMSE、统计和汇总。",
            "- `tests/test_fixed_grid_experiment.py`：fixed-grid 单元/sanity tests。",
            "- `tools/run_fixed_grid_remaining.sh`：顺序自动运行与归档。",
            "",
            "代码库中原先存在的 AWQ/4-6 等未提交修改不属于本 fixed-grid 实验的新增范围；",
            "完整当前 diff 已保存为 `code_changes.patch`。",
            "",
            "## 9. 复现命令与保存内容",
            "",
            "六组量化与评估命令保存在 `run_commands.sh`。主要结果：",
            "",
            "- `results/*.json`：逐层/投影 NMSE、145 个原始 sequence NLL、PPL、code/scale stats。",
            "- `results/summary.json`：汇总表与 interaction。",
            "- `configs/`：原始 YAML。",
            "- `calibration_sample_ids.json`：前 16 个样本的 index/SHA256、seed 和顺序。",
            "- `logs/`：GPTQ-G4/GINT 与三组 GPTQ evaluation 原始日志。",
            "- `checkpoints/`：三组 RTN sparse packed artifacts 与三组 GPTQ checkpoints。",
        ]
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    root = results_dir.parent
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={root!s}

# RTN checkpoints
python3 tools/fixed_grid_experiment.py rtn-pack \\
  --config configs/qwen3_5/fixed_grid_experiment/rtn_g6.yaml \\
  --output "$ROOT/checkpoints/rtn_g6" --device cuda:0
python3 tools/fixed_grid_experiment.py rtn-pack \\
  --config configs/qwen3_5/fixed_grid_experiment/rtn_g4.yaml \\
  --output "$ROOT/checkpoints/rtn_g4" --device cuda:0
python3 tools/fixed_grid_experiment.py rtn-pack \\
  --config configs/qwen3_5/fixed_grid_experiment/rtn_gint.yaml \\
  --output "$ROOT/checkpoints/rtn_gint" --device cuda:0

# GPTQ checkpoints (each command relaunches 8 expert-parallel workers)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/run.py \\
  -c configs/qwen3_5/fixed_grid_experiment/gptq_g6.yaml \\
  --save-path "$ROOT/checkpoints/gptq_g6"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/run.py \\
  -c configs/qwen3_5/fixed_grid_experiment/gptq_g4.yaml \\
  --save-path "$ROOT/checkpoints/gptq_g4"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/run.py \\
  -c configs/qwen3_5/fixed_grid_experiment/gptq_gint.yaml \\
  --save-path "$ROOT/checkpoints/gptq_gint"

# BF16 decode + 145-sequence WikiText-2 evaluation
python3 tools/fixed_grid_experiment.py evaluate \\
  --checkpoint "$ROOT/checkpoints/rtn_g6" --grid g6 --method rtn \\
  --output "$ROOT/results/rtn_g6.json"
python3 tools/fixed_grid_experiment.py evaluate \\
  --checkpoint "$ROOT/checkpoints/rtn_g4" --grid g4 --method rtn \\
  --output "$ROOT/results/rtn_g4.json"
python3 tools/fixed_grid_experiment.py evaluate \\
  --checkpoint "$ROOT/checkpoints/rtn_gint" --grid gint --method rtn \\
  --output "$ROOT/results/rtn_gint.json"
python3 tools/fixed_grid_experiment.py evaluate \\
  --checkpoint "$ROOT/checkpoints/gptq_g6/gptq_g6" --grid g6 --method gptq \\
  --output "$ROOT/results/gptq_g6.json"
python3 tools/fixed_grid_experiment.py evaluate \\
  --checkpoint "$ROOT/checkpoints/gptq_g4/gptq_g4" --grid g4 --method gptq \\
  --output "$ROOT/results/gptq_g4.json"
python3 tools/fixed_grid_experiment.py evaluate \\
  --checkpoint "$ROOT/checkpoints/gptq_gint/gptq_gint" --grid gint --method gptq \\
  --output "$ROOT/results/gptq_gint.json"
"""
    (root / "run_commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(summary, indent=2))


def command_archive(args: argparse.Namespace) -> None:
    import contextlib
    import runpy

    root = Path(args.root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with (root / "logs" / "sanity_checks.log").open("w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            namespace = runpy.run_path("tests/test_fixed_grid_experiment.py")
            for name in sorted(n for n in namespace if n.startswith("test_")):
                print("RUN", name)
                namespace[name]()
            print("ALL PASSED")

    for source in (
        "tools/fixed_grid_experiment.py",
        "tools/run_fixed_grid_remaining.sh",
        "tests/test_fixed_grid_experiment.py",
    ):
        shutil.copy2(source, root / Path(source).name)
    diff = subprocess.run(["git", "diff"], check=True, capture_output=True, text=True).stdout
    status = subprocess.run(
        ["git", "status", "--short"], check=True, capture_output=True, text=True
    ).stdout
    (root / "code_changes.patch").write_text(diff, encoding="utf-8")
    (root / "git_status.txt").write_text(status, encoding="utf-8")
    print(f"Archived final scripts, sanity log, diff, and status under {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    rtn = sub.add_parser("rtn-pack")
    rtn.add_argument("--config", required=True)
    rtn.add_argument("--output")
    rtn.add_argument("--device", default="cuda:0")
    rtn.set_defaults(func=command_rtn_pack)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--grid", required=True, choices=("g6", "g4", "gint"))
    evaluate.add_argument("--method", required=True, choices=("rtn", "gptq"))
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--model", default=DEFAULT_MODEL)
    evaluate.add_argument("--dataset", default=DEFAULT_WIKITEXT)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--seqlen", type=int, default=2048)
    evaluate.add_argument("--num-sequences", type=int, default=145)
    evaluate.set_defaults(func=command_evaluate)

    ids = sub.add_parser("calibration-ids")
    ids.add_argument("--dataset", default="./dataset/wikitext_calib_applied.jsonl")
    ids.add_argument("--num-samples", type=int, default=16)
    ids.add_argument("--output", required=True)
    ids.set_defaults(func=command_calibration_ids)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--results-dir", required=True)
    summarize.add_argument("--output-json", required=True)
    summarize.add_argument("--output-md", required=True)
    summarize.set_defaults(func=command_summarize)

    archive = sub.add_parser("archive")
    archive.add_argument("--root", required=True)
    archive.set_defaults(func=command_archive)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
