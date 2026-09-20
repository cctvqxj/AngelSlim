#!/usr/bin/env python3
"""将输入模型 config.json 的指定字段同步为 HY4 FP8 参考模型的值。

默认同步字段：
  - model_type
  - num_key_value_heads
  - layer_types（attention 类型命名）
  - use_cache

``--reference`` 必须显式传入，不在源码中保存任何模型路径。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

SYNC_FIELDS = (
    "model_type",
    "num_key_value_heads",
    "layer_types",
    "use_cache",
)

SKIP_AUXILIARY_FILES = {
    ".encrypted",
    "angelslim_config.json",
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "model_cache_info_taiji.json",
    "pytorch_model.bin.index.json",
}

SKIP_AUXILIARY_DIRS = {
    ".expert_parallel_manifests",
    ".git",
    "__pycache__",
}

SINGLE_FILE_MODEL_WEIGHTS = {
    "model.safetensors",
    "pytorch_model.bin",
}

ANGELSLIM_SIDECAR_NAME = "angelslim_config.json"
PRIVATE_CONFIG_FIELDS = {
    "mtp_quant_algo",
    "mtp_quant_method",
}


def resolve_config_path(path: Path) -> Path:
    """允许传入模型目录或 config.json 文件。"""
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 config.json：{path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} 的顶层 JSON 不是对象")
    return value


def short_value(value: Any) -> str:
    if isinstance(value, list):
        counts: dict[str, int] = {}
        for item in value:
            key = repr(item)
            counts[key] = counts.get(key, 0) + 1
        distribution = ", ".join(f"{name} × {count}" for name, count in sorted(counts.items()))
        return f"list(len={len(value)}; {distribution})"
    return repr(value)


def validate_layer_types(
    input_config: dict[str, Any],
    reference_config: dict[str, Any],
    force: bool,
) -> None:
    reference_layers = reference_config.get("layer_types")
    if not isinstance(reference_layers, list):
        raise TypeError("参考 config 的 layer_types 不是列表")

    input_hidden_layers = input_config.get("num_hidden_layers")
    reference_hidden_layers = reference_config.get("num_hidden_layers")
    if len(reference_layers) != reference_hidden_layers:
        raise ValueError(
            "参考 config 不一致："
            f"len(layer_types)={len(reference_layers)}，"
            f"num_hidden_layers={reference_hidden_layers}"
        )

    if (
        input_hidden_layers is not None
        and input_hidden_layers != reference_hidden_layers
        and not force
    ):
        raise ValueError(
            "输入模型和参考模型的 num_hidden_layers 不一致："
            f"{input_hidden_layers} != {reference_hidden_layers}。"
            "如果确认要强制复制参考 layer_types，请加 --force。"
        )


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            shutil.copystat(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def split_angelslim_quant_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, Any]]]:
    """从 HF ``config.json`` 中移出 AngelSlim 私有量化字段。

    标准 ``quantization_config`` 仍保留在 ``config.json``，因为 Transformers
    和推理框架需要它。名称中包含 ``angelslim`` 的顶层字段，以及 MTP
    保存策略字段，则迁移至 ``angelslim_config.json`` 的算法 sidecar。
    """
    cleaned = copy.deepcopy(config)
    quant_updates: dict[str, Any] = {}
    private_model_config: dict[str, Any] = {}
    moved: list[tuple[str, Any]] = []

    for field in list(cleaned):
        if "angelslim" in field.lower():
            value = cleaned.pop(field)
            private_model_config[field] = value
            moved.append((field, value))

    for field in PRIVATE_CONFIG_FIELDS:
        if field in cleaned:
            value = cleaned.pop(field)
            quant_updates[field] = value
            moved.append((field, value))

    if moved:
        quant_updates["algorithm"] = "rtn"
    if private_model_config:
        quant_updates["private_model_config"] = private_model_config
    return cleaned, quant_updates, moved


def merge_angelslim_quant_config(
    sidecar: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """将迁移出的字段合并到 AngelSlim sidecar，保留已有完整配置。"""
    merged = copy.deepcopy(sidecar)
    if not updates:
        return merged

    config_key = "rtn_config"
    quant_config = merged.get(config_key)
    if not isinstance(quant_config, dict):
        quant_config = {}
    else:
        quant_config = copy.deepcopy(quant_config)

    private_updates = updates.get("private_model_config")
    if isinstance(private_updates, dict):
        private_model_config = quant_config.get("private_model_config")
        if not isinstance(private_model_config, dict):
            private_model_config = {}
        else:
            private_model_config = copy.deepcopy(private_model_config)
        private_model_config.update(copy.deepcopy(private_updates))
        quant_config["private_model_config"] = private_model_config

    for field, value in updates.items():
        if field != "private_model_config":
            quant_config[field] = copy.deepcopy(value)
    merged[config_key] = quant_config
    return merged


def source_weight_files(source_model: Path) -> set[str]:
    """从模型索引读取权重分片名，避免把源权重复制到目标模型。"""
    weights = set(SINGLE_FILE_MODEL_WEIGHTS)
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = source_model / index_name
        if not index_path.is_file():
            continue
        try:
            index = load_json(index_path)
            weight_map = index.get("weight_map", {})
            if isinstance(weight_map, dict):
                weights.update(
                    Path(filename).as_posix()
                    for filename in weight_map.values()
                    if isinstance(filename, str)
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # 辅助文件复制不应因为损坏的源权重索引而完全失败。
            # 索引文件自身仍在 SKIP_AUXILIARY_FILES 中，不会被复制。
            pass
    return weights


def find_missing_auxiliary_files(
    source_model: Path,
    target_model: Path,
) -> list[tuple[Path, Path, Path]]:
    """返回需要复制的 ``(相对路径, 源路径, 目标路径)``。

    只复制目标中不存在的非权重文件，不覆盖量化模型已经生成的配置、
    tokenizer、processor 或其他资源。
    """
    source_model = source_model.expanduser().resolve()
    target_model = target_model.expanduser().resolve()
    if not source_model.is_dir():
        raise FileNotFoundError(f"辅助文件源模型目录不存在：{source_model}")
    if not target_model.is_dir():
        raise FileNotFoundError(f"目标模型目录不存在：{target_model}")
    if source_model == target_model:
        return []

    weights = source_weight_files(source_model)
    missing: list[tuple[Path, Path, Path]] = []
    for root, dirnames, filenames in os.walk(source_model, followlinks=False):
        dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in SKIP_AUXILIARY_DIRS)
        root_path = Path(root)
        for filename in sorted(filenames):
            source_file = root_path / filename
            relative_file = source_file.relative_to(source_model)
            relative_name = relative_file.as_posix()
            if filename in SKIP_AUXILIARY_FILES:
                continue
            if filename.endswith((".pyc", ".pyo")):
                continue
            if relative_name in weights:
                continue

            target_file = target_model / relative_file
            if os.path.lexists(target_file):
                continue
            missing.append((relative_file, source_file, target_file))
    return missing


def copy_missing_auxiliary_files(
    files: list[tuple[Path, Path, Path]],
) -> list[Path]:
    """复制 ``find_missing_auxiliary_files`` 找到的缺失辅助文件。"""
    copied: list[Path] = []
    for relative_file, source_file, target_file in files:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if source_file.is_symlink():
            target_file.symlink_to(os.readlink(source_file))
        else:
            shutil.copy2(source_file, target_file)
        copied.append(relative_file)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "将输入模型 config.json 中的 model_type、"
            "num_key_value_heads、layer_types 和 use_cache "
            "同步为 HY4 FP8 参考模型的值。"
        )
    )
    parser.add_argument(
        "input_model",
        type=Path,
        help="输入模型目录，或者 config.json 文件路径",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="参考 BF16 模型目录或 config.json",
    )
    parser.add_argument(
        "--aux-source",
        type=Path,
        help=(
            "辅助文件来源模型目录或 config.json。默认使用 --reference 所在目录；"
            "如需从原始 BF16 模型复制 .py、processor config 等文件，可显式指定。"
        ),
    )
    parser.add_argument(
        "--no-copy-files",
        action="store_true",
        help="只同步 config 字段，不复制缺失的辅助文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要修改的字段，不写文件",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 num_hidden_layers 不一致，也强制复制参考 layer_types",
    )
    args = parser.parse_args()

    input_path = resolve_config_path(args.input_model)
    reference_path = resolve_config_path(args.reference)
    input_model_path = input_path.parent
    auxiliary_source_path = (
        resolve_config_path(args.aux_source).parent
        if args.aux_source is not None
        else reference_path.parent
    )

    if input_path == reference_path:
        raise ValueError("输入 config 和参考 config 是同一个文件，无需修改")

    input_config = load_json(input_path)
    reference_config = load_json(reference_path)
    sidecar_path = input_model_path / ANGELSLIM_SIDECAR_NAME
    input_sidecar = load_json(sidecar_path) if sidecar_path.is_file() else {}

    missing_fields = [field for field in SYNC_FIELDS if field not in reference_config]
    if missing_fields:
        raise KeyError(f"参考 config 缺少待同步字段：{', '.join(missing_fields)}")

    validate_layer_types(input_config, reference_config, args.force)

    updated_config = copy.deepcopy(input_config)
    changes: list[tuple[str, Any, Any]] = []
    for field in SYNC_FIELDS:
        old_value = input_config.get(field, "<字段不存在>")
        new_value = copy.deepcopy(reference_config[field])
        if old_value != new_value:
            changes.append((field, old_value, new_value))
        updated_config[field] = new_value

    updated_config, quant_updates, moved_private_fields = split_angelslim_quant_config(
        updated_config,
    )
    updated_sidecar = merge_angelslim_quant_config(
        input_sidecar,
        quant_updates,
    )
    config_changed = bool(changes or moved_private_fields)
    sidecar_changed = updated_sidecar != input_sidecar

    auxiliary_files = (
        []
        if args.no_copy_files
        else find_missing_auxiliary_files(
            auxiliary_source_path,
            input_model_path,
        )
    )

    print(f"输入配置：{input_path}")
    print(f"参考配置：{reference_path}")
    if changes:
        print("将修改以下字段：")
        for field, old_value, new_value in changes:
            print(f"  {field}")
            print(f"    原值：{short_value(old_value)}")
            print(f"    新值：{short_value(new_value)}")
    else:
        print("指定字段已经与 FP8 参考配置一致。")

    if moved_private_fields:
        print(
            "将从 config.json 迁移到 "
            f"{ANGELSLIM_SIDECAR_NAME}.rtn_config 的字段："
        )
        for field, value in moved_private_fields:
            print(f"  {field}: {short_value(value)}")

    if not args.no_copy_files:
        print(f"辅助文件来源：{auxiliary_source_path}")
        if auxiliary_files:
            print(f"将复制 {len(auxiliary_files)} 个缺失的辅助文件：")
            for relative_file, _, _ in auxiliary_files:
                print(f"  {relative_file.as_posix()}")
        else:
            print("目标模型不缺少辅助文件。")

    if not config_changed and not sidecar_changed and not auxiliary_files:
        print("无需修改或复制。")
        return 0

    if args.dry_run:
        print("dry-run：未写入文件。")
        return 0

    if config_changed:
        atomic_write_json(input_path, updated_config)

        # 写入后重新读取并验证，避免部分写入或字段遗漏。
        written_config = load_json(input_path)
        for field in SYNC_FIELDS:
            if written_config.get(field) != reference_config[field]:
                raise RuntimeError(f"写入后字段校验失败：{field}")
        private_fields_left = [
            field
            for field in written_config
            if "angelslim" in field.lower()
            or field in PRIVATE_CONFIG_FIELDS
        ]
        if private_fields_left:
            raise RuntimeError(
                "写入后 config.json 仍包含 AngelSlim 私有字段："
                + ", ".join(sorted(private_fields_left))
            )

        print(f"配置修改完成：{input_path}")
    if sidecar_changed:
        atomic_write_json(sidecar_path, updated_sidecar)
        written_sidecar = load_json(sidecar_path)
        config_key = "rtn_config"
        if written_sidecar.get(config_key) != updated_sidecar.get(config_key):
            raise RuntimeError(
                f"写入后 angelslim_config.json.{config_key} 校验失败"
            )
        print(f"AngelSlim 配置更新完成：{sidecar_path}")
    copied_files = copy_missing_auxiliary_files(auxiliary_files)
    if copied_files:
        print(f"辅助文件复制完成：{len(copied_files)} 个")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
