import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

from tools.hy4_mxfp4_rtn_to_fp8_ue8m0 import (
    build_fp8_config,
    is_fp8_weight,
    main as build_mixed_checkpoint,
)
from tools.hy4_mxfp4_weight_only import main as build_rtn_checkpoint
from tools.validate_hy4_mxfp4_rtn import main as validate_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_checkpoint(path: Path) -> None:
    path.mkdir()
    config = {
        "model_type": "hy_v4",
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "mlp_layer_types": ["dense", "sparse"],
        "layer_types": ["full_attention", "sparse_attention"],
        "use_cache": False,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    tensors = {
        "model.layers.1.mlp.experts.gate_up_proj": torch.randn(
            2, 64, 32, dtype=torch.bfloat16
        ),
        "model.layers.1.mlp.experts.down_proj": torch.randn(
            2, 32, 32, dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            32, 32, dtype=torch.bfloat16
        ),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.randn(
            32, 32, dtype=torch.bfloat16
        ),
        "model.layers.1.self_attn.indexer.wk.weight": torch.randn(
            32, 32, dtype=torch.bfloat16
        ),
        "model.layers.1.mlp.gate.weight": torch.randn(
            2, 32, dtype=torch.bfloat16
        ),
        "model.mtp_layers.0.mlp.experts.gate_up_proj": torch.randn(
            2, 64, 32, dtype=torch.bfloat16
        ),
        "model.mtp_layers.0.mlp.experts.down_proj": torch.randn(
            2, 32, 32, dtype=torch.bfloat16
        ),
        "lm_head.weight": torch.randn(16, 32, dtype=torch.bfloat16),
    }
    shard = "model.safetensors"
    save_file(tensors, path / shard, metadata={"format": "pt"})
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard for name in tensors},
            }
        ),
        encoding="utf-8",
    )


def _read_tensor(model_path: Path, name: str) -> torch.Tensor:
    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    with safe_open(
        model_path / index["weight_map"][name],
        framework="pt",
        device="cpu",
    ) as reader:
        return reader.get_tensor(name)


def test_fp8_selection_and_config():
    assert is_fp8_weight("model.layers.0.mlp.gate_proj.weight")
    assert is_fp8_weight("model.layers.1.mlp.shared_experts.gate_proj.weight")
    assert is_fp8_weight("model.layers.1.self_attn.indexer.wk.weight")
    assert is_fp8_weight("model.mtp_layers.0.self_attn.q_a_proj.weight")
    assert not is_fp8_weight("model.layers.1.mlp.gate.weight")
    assert not is_fp8_weight("lm_head.weight")

    config = build_fp8_config(["lm_head", "model.embed_tokens"])
    assert config["quant_method"] == "fp8"
    assert config["weight_block_size"] == [128, 128]
    assert config["scale_fmt"] == "ue8m0"


def test_end_to_end_rtn_pipeline(tmp_path):
    source = tmp_path / "source"
    stage1 = tmp_path / "stage1"
    output = tmp_path / "output"
    _write_checkpoint(source)

    include_patterns = [
        r"^model\.layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj)$"
    ]
    exclude_patterns = [r"^model\.mtp_layers\..*"]
    build_rtn_checkpoint(
        input_path=str(source),
        output_path=str(stage1),
        num_workers=1,
        use_gpu=False,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    build_mixed_checkpoint(
        input_path=str(stage1),
        output_path=str(output),
        num_workers=1,
        use_gpu=False,
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "sync_hy4_rtn_config_to_fp8.py"),
            str(output),
            "--reference",
            str(source),
            "--aux-source",
            str(source),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    validate_checkpoint(
        stage1_path=str(stage1),
        output_path=str(output),
        bf16_path=str(source),
        sample_count=12,
    )

    expert = "model.layers.1.mlp.experts.0.gate_proj"
    assert _read_tensor(output, f"{expert}.weight").dtype == torch.uint8
    assert _read_tensor(output, f"{expert}.weight_scale").dtype == torch.uint8
    assert _read_tensor(
        output, "model.layers.0.mlp.gate_proj.weight"
    ).dtype == torch.float8_e4m3fn
    assert _read_tensor(
        output, "model.mtp_layers.0.mlp.experts.0.gate_proj.weight"
    ).dtype == torch.float8_e4m3fn
    assert _read_tensor(output, "lm_head.weight").dtype == torch.bfloat16

    final_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert "angelslim_mxfp4_config" not in final_config
    assert "angelslim_mixed_mxfp4_fp8_config" not in final_config
    assert "mtp_quant_algo" not in final_config
    sidecar = json.loads(
        (output / "angelslim_config.json").read_text(encoding="utf-8")
    )
    assert sidecar["rtn_config"]["algorithm"] == "rtn"
    assert sidecar["rtn_config"]["mtp_quant_algo"] == "FP8"
