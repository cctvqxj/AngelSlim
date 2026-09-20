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

"""Streaming MXFP4 weight-only conversion for HY4 routed experts.

The converter operates directly on Hugging Face safetensor shards and never
constructs the full HY4 model.  Selected 2-D Linear weights are converted to
OCP MXFP4:

* E2M1 values, packed two 4-bit values per ``uint8``;
* one E8M0 ``uint8`` scale per 32 values along the input dimension;
* no activation scale and no global weight scale.

HY4 routed experts are stored as fused 3-D tensors:

* ``experts.gate_up_proj``: ``[E, 2I, H]``;
* ``experts.down_proj``: ``[E, H, I]``.

When selected, they are split into per-expert ``gate_proj`` / ``up_proj`` /
``down_proj`` checkpoint keys.  Scheme A deliberately selects only main-model
routed experts; MTP and all other tensors pass through unchanged as BF16/FP32.
"""

import json
import multiprocessing as mp
import os
import re
import shutil
from argparse import ArgumentParser

import torch
import yaml
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

MXFP4_GROUP_SIZE = 32
MXFP4_E8M0_BIAS = 127
MXFP4_E2M1_MAX = 6.0
MXFP4_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_FUSED_EXPERT_PATTERN = re.compile(
    r"^(model\.(?:layers|mtp_layers)\.\d+\.mlp\.experts)" r"\.(gate_up_proj|down_proj)$"
)


def compile_patterns(patterns, field_name):
    compiled = []
    for pattern in patterns or []:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid {field_name} regex {pattern!r}: {exc}") from exc
    return compiled


def module_name_from_weight_name(weight_name):
    return weight_name[: -len(".weight")] if weight_name.endswith(".weight") else weight_name


def should_quantize(weight_name, include_patterns, exclude_patterns):
    module_name = module_name_from_weight_name(weight_name)
    return any(pattern.fullmatch(module_name) for pattern in include_patterns) and not any(
        pattern.fullmatch(module_name) for pattern in exclude_patterns
    )


def _mxfp4_cast_to_e2m1(weight):
    bounds = MXFP4_E2M1_BOUNDS.to(weight.device)
    tie_round_up_mask = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0],
        dtype=torch.uint8,
        device=weight.device,
    )
    tie_round_up_mask = tie_round_up_mask.expand(*weight.shape, 7)

    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ordinal = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    round_up = torch.any(
        (weight_abs.unsqueeze(-1) == bounds) * tie_round_up_mask,
        dim=-1,
    )
    return (sign_bit * 0b1000 + ordinal + round_up).to(torch.uint8)


@torch.no_grad()
def mxfp4_rtn_pack(weight, group_size=MXFP4_GROUP_SIZE):
    """Pack a 2-D weight into MXFP4 E2M1 values and E8M0 block scales."""
    if weight.ndim != 2:
        raise ValueError(
            f"MXFP4 weight-only expects a 2-D weight, got shape {tuple(weight.shape)}"
        )
    if group_size != MXFP4_GROUP_SIZE:
        raise ValueError(f"MXFP4 group_size must be {MXFP4_GROUP_SIZE}, got {group_size}.")
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"in_features={weight.shape[-1]} is not divisible by " f"MXFP4 group_size={group_size}"
        )

    blocks = weight.float().reshape(weight.shape[0], -1, group_size)
    block_amax = blocks.abs().amax(dim=-1)
    scale_target = (block_amax / MXFP4_E2M1_MAX).clamp_min(1e-30)
    decoded_scale = torch.pow(2.0, torch.ceil(torch.log2(scale_target)))
    decoded_scale = torch.where(
        block_amax == 0,
        torch.ones_like(decoded_scale),
        decoded_scale,
    )

    scaled = (blocks / decoded_scale.unsqueeze(-1)).reshape(weight.shape)
    codes = _mxfp4_cast_to_e2m1(scaled)
    packed_weight = ((codes[:, 1::2] << 4) | codes[:, 0::2]).contiguous()
    # Keep E8M0 exponent codes as uint8 bytes in the checkpoint.  vLLM's
    # MXFP4 MoE parameters are uint8, and some HYV4 loaders copy checkpoint
    # tensors without first reinterpreting float8_e8m0fnu as uint8.  Saving
    # the semantic float8 dtype would therefore numerically cast sub-unit
    # scales to zero during loading instead of preserving their raw bytes.
    encoded_scale = (torch.log2(decoded_scale.float()) + MXFP4_E8M0_BIAS).to(
        torch.uint8
    )
    return packed_weight.cpu(), encoded_scale.cpu()


def quantized_names(module_name):
    return {
        "weight": f"{module_name}.weight",
        "weight_scale": f"{module_name}.weight_scale",
    }


def quantize_and_store(state_dict, index, module_name, weight, file_name, group_size):
    packed_weight, weight_scale = mxfp4_rtn_pack(weight, group_size=group_size)
    names = quantized_names(module_name)
    state_dict[names["weight"]] = packed_weight
    state_dict[names["weight_scale"]] = weight_scale
    for name in names.values():
        index[name] = file_name


def quantize_fused_experts(
    state_dict,
    index,
    weight_name,
    weight,
    file_name,
    group_size,
    device,
):
    match = _FUSED_EXPERT_PATTERN.fullmatch(weight_name)
    if match is None:
        return False
    if weight.ndim != 3:
        raise ValueError(
            f"Expected fused HY4 expert tensor {weight_name} to be 3-D, "
            f"got {tuple(weight.shape)}"
        )

    prefix, projection = match.groups()
    for expert_idx in range(weight.shape[0]):
        if projection == "gate_up_proj":
            gate, up = weight[expert_idx].chunk(2, dim=0)
            projections = (("gate_proj", gate), ("up_proj", up))
        else:
            projections = (("down_proj", weight[expert_idx]),)

        for projection_name, expert_weight in projections:
            module_name = f"{prefix}.{expert_idx}.{projection_name}"
            quantize_and_store(
                state_dict,
                index,
                module_name,
                expert_weight.to(device),
                file_name,
                group_size,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return True


def process_shard(
    worker_id,
    file_name,
    input_path,
    output_path,
    group_size,
    use_gpu,
    include_pattern_strings,
    exclude_pattern_strings,
):
    include_patterns = compile_patterns(include_pattern_strings, "include_patterns")
    exclude_patterns = compile_patterns(exclude_pattern_strings, "exclude_patterns")
    if use_gpu:
        device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    state_dict = {}
    index = {}
    quantized_modules = 0
    with safe_open(os.path.join(input_path, file_name), framework="pt", device="cpu") as reader:
        for weight_name in reader.keys():
            tensor = reader.get_tensor(weight_name)
            if not should_quantize(weight_name, include_patterns, exclude_patterns):
                state_dict[weight_name] = tensor
                index[weight_name] = file_name
                continue

            if quantize_fused_experts(
                state_dict,
                index,
                weight_name,
                tensor,
                file_name,
                group_size,
                device,
            ):
                match = _FUSED_EXPERT_PATTERN.fullmatch(weight_name)
                quantized_modules += tensor.shape[0] * (
                    2 if match.group(2) == "gate_up_proj" else 1
                )
            else:
                module_name = module_name_from_weight_name(weight_name)
                quantize_and_store(
                    state_dict,
                    index,
                    module_name,
                    tensor.to(device),
                    file_name,
                    group_size,
                )
                quantized_modules += 1

            del tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    shard_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    save_file(
        state_dict,
        os.path.join(output_path, file_name),
        metadata={"format": "pt"},
    )
    return index, quantized_modules, shard_size


def worker(
    worker_id,
    file_names,
    input_path,
    output_path,
    group_size,
    use_gpu,
    include_patterns,
    exclude_patterns,
    return_dict,
):
    local_index = {}
    local_count = 0
    local_size = 0
    for file_name in tqdm(file_names, desc=f"HY4 MXFP4 weight-only worker {worker_id}"):
        shard_index, count, shard_size = process_shard(
            worker_id,
            file_name,
            input_path,
            output_path,
            group_size,
            use_gpu,
            include_patterns,
            exclude_patterns,
        )
        overlap = set(local_index).intersection(shard_index)
        if overlap:
            raise RuntimeError(
                f"Duplicate checkpoint keys while processing {file_name}: "
                f"{sorted(overlap)[:10]}"
            )
        local_index.update(shard_index)
        local_count += count
        local_size += shard_size
    return_dict[worker_id] = {
        "index": local_index,
        "count": local_count,
        "total_size": local_size,
    }


def collect_shards(input_path):
    index_path = os.path.join(input_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    if os.path.isfile(os.path.join(input_path, "model.safetensors")):
        return ["model.safetensors"]
    raise FileNotFoundError(f"No safetensors checkpoint found under {input_path}")


def prepare_output_directory(output_path, input_path):
    input_real = os.path.realpath(input_path)
    output_real = os.path.realpath(output_path)
    try:
        output_is_nested = os.path.commonpath([input_real, output_real]) == input_real
    except ValueError:
        output_is_nested = False
    if output_real == input_real or output_is_nested:
        raise ValueError("output_path must be outside the input checkpoint directory")
    if os.path.exists(output_path) and os.listdir(output_path):
        raise ValueError(f"output_path must be empty: {output_path}")
    os.makedirs(output_path, exist_ok=True)


def copy_auxiliary_files(input_path, output_path):
    generated = {
        "angelslim_config.json",
        "config.json",
        "expert_quantization_report.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
        "model.safetensors",
    }
    for name in os.listdir(input_path):
        if name in generated or name.endswith(".safetensors"):
            continue
        src = os.path.join(input_path, name)
        dst = os.path.join(output_path, name)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _regex_target(pattern):
    return pattern if pattern.startswith("re:") else f"re:{pattern}"


def expand_quantized_targets(include_patterns):
    """Map source fused-expert selectors to the emitted split module names."""
    targets = []
    for pattern in include_patterns:
        if "experts" in pattern and "gate_up_proj|down_proj" in pattern:
            targets.append(
                re.sub(
                    r"\(\?:gate_up_proj\|down_proj\)",
                    r"\\d+\\.(?:gate_proj|up_proj|down_proj)",
                    pattern,
                )
            )
        else:
            targets.append(pattern)
    return targets


def build_quantization_config(group_size, include_patterns, exclude_patterns):
    targets = [_regex_target(pattern) for pattern in expand_quantized_targets(include_patterns)]
    ignore = [_regex_target(pattern) for pattern in exclude_patterns]
    return {
        "quant_method": "compressed-tensors",
        "ignore": ignore,
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": group_size,
                    "symmetric": True,
                    "dynamic": False,
                    "type": "float",
                    "scale_dtype": "uint8",
                },
                "input_activations": None,
                "output_activations": None,
                "targets": targets,
            }
        },
        "kv_cache_scheme": None,
        "format": "mxfp4-pack-quantized",
        "quantization_status": "compressed",
        "scale_fmt": "ue8m0",
    }


def build_hf_quant_config(
    group_size,
    include_patterns,
    exclude_patterns,
    config_groups=None,
):
    # Keep this as informational metadata only. The authoritative mixed-
    # precision selector is config.json's compressed-tensors config_groups.
    # Publishing a broad ModelOpt ``quant_algo=MXFP4`` here would incorrectly
    # claim that attention/shared/dense linears were quantized as well.
    return {
        "angelslim_mxfp4_weight_only": {
            "quant_algo": "MXFP4",
            "group_size": group_size,
            "kv_cache_quant_algo": None,
            "exclude_modules": list(exclude_patterns),
            "include_patterns": list(include_patterns),
            "config_groups": config_groups or {},
        }
    }


def main(
    input_path,
    output_path,
    group_size=MXFP4_GROUP_SIZE,
    num_workers=8,
    use_gpu=True,
    include_patterns=None,
    exclude_patterns=None,
):
    include_patterns = list(include_patterns or [])
    exclude_patterns = list(exclude_patterns or [])
    if not include_patterns:
        raise ValueError("MXFP4 weight-only requires at least one include_patterns regex")
    if group_size != MXFP4_GROUP_SIZE:
        raise ValueError(f"MXFP4 group_size must be {MXFP4_GROUP_SIZE}, got {group_size}.")
    compile_patterns(include_patterns, "include_patterns")
    compile_patterns(exclude_patterns, "exclude_patterns")

    config_path = os.path.join(input_path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if "quantization_config" in config:
        raise AssertionError("MXFP4 weight-only expects an unquantized BF16 source checkpoint")

    if use_gpu and torch.cuda.device_count() == 0:
        print(
            "[HY4 MXFP4 weight-only] No CUDA device found; falling back to CPU.",
            flush=True,
        )
        use_gpu = False

    prepare_output_directory(output_path, input_path)
    shards = collect_shards(input_path)
    num_workers = max(1, min(int(num_workers), len(shards)))
    subsets = [shards[i::num_workers] for i in range(num_workers)]

    if num_workers == 1:
        return_dict = {}
        worker(
            0,
            subsets[0],
            input_path,
            output_path,
            group_size,
            use_gpu,
            include_patterns,
            exclude_patterns,
            return_dict,
        )
    else:
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        for worker_id in range(num_workers):
            process = mp.Process(
                target=worker,
                args=(
                    worker_id,
                    subsets[worker_id],
                    input_path,
                    output_path,
                    group_size,
                    use_gpu,
                    include_patterns,
                    exclude_patterns,
                    return_dict,
                ),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"MXFP4 weight-only worker pid={process.pid} failed: " f"{process.exitcode}"
                )

    weight_map = {}
    quantized_count = 0
    total_size = 0
    for result in return_dict.values():
        overlap = set(weight_map).intersection(result["index"])
        if overlap:
            raise RuntimeError(
                "Duplicate checkpoint keys across MXFP4 workers: " f"{sorted(overlap)[:10]}"
            )
        weight_map.update(result["index"])
        quantized_count += result["count"]
        total_size += result["total_size"]

    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": dict(sorted(weight_map.items())),
            },
            f,
            indent=2,
        )

    copy_auxiliary_files(input_path, output_path)
    config["quantization_config"] = build_quantization_config(
        group_size,
        include_patterns,
        exclude_patterns,
    )
    # Scheme A intentionally leaves MTP untouched.
    config["mtp_quant_algo"] = "bf16"
    config["angelslim_mxfp4_config"] = {
        "algorithm": "rtn",
        "weight_format": "mxfp4",
        "scale_format": "ue8m0",
        "group_size": group_size,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "quantized_module_count": quantized_count,
        "mtp_quant_algo": "bf16",
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with open(os.path.join(output_path, "hf_quant_config.json"), "w") as f:
        json.dump(
            build_hf_quant_config(
                group_size,
                include_patterns,
                exclude_patterns,
                config_groups=config["quantization_config"]["config_groups"],
            ),
            f,
            indent=4,
        )
        f.write("\n")

    print(
        f"[HY4 MXFP4 weight-only] Quantized {quantized_count} modules into " f"{output_path}",
        flush=True,
    )


def load_yaml_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    quant = config["compression"]["quantization"]
    method = quant.get("quant_method", {})
    return {
        "group_size": method.get("group_size", MXFP4_GROUP_SIZE),
        "num_workers": method.get("num_workers", 8),
        "use_gpu": method.get("use_gpu", True),
        "include_patterns": method.get("include_patterns", []),
        "exclude_patterns": method.get("exclude_patterns", []),
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--group-size", type=int, default=MXFP4_GROUP_SIZE)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--include-pattern", action="append", default=[])
    parser.add_argument("--exclude-pattern", action="append", default=[])
    args = parser.parse_args()

    if not args.input_path or not args.output_path:
        parser.error("--input-path and --output-path are required")
    kwargs = load_yaml_config(args.config) if args.config else {}
    kwargs.update(
        {
            "input_path": args.input_path,
            "output_path": args.output_path,
        }
    )
    if not args.config:
        kwargs.update(
            {
                "group_size": args.group_size,
                "num_workers": args.num_workers,
                "use_gpu": not args.cpu,
                "include_patterns": args.include_pattern,
                "exclude_patterns": args.exclude_pattern,
            }
        )
    elif args.cpu:
        kwargs["use_gpu"] = False
    if args.num_workers is not None:
        kwargs["num_workers"] = args.num_workers
    main(**kwargs)
