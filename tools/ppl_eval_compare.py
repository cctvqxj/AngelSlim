"""PPL eval for NVFP4-GPTQ quantized models using AngelSlim's QDQ module."""

import glob
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def nvfp4_dequantize(packed_weight, weight_scale, weight_scale_2, block_size=16):
    """Dequantize NVFP4 packed uint8 weight to bf16."""
    dtype = torch.bfloat16
    out_cols = packed_weight.shape[1] * 2
    unpacked = torch.empty(packed_weight.shape[0], out_cols, dtype=dtype)
    unpacked[..., 1::2] = packed_weight >> 4
    unpacked[..., 0::2] = packed_weight & 0x0F
    shape = unpacked.shape
    unpacked = E2M1_VALUES[unpacked.reshape(-1).long()].reshape(shape)

    scale = weight_scale.to(torch.float32) * weight_scale_2.to(torch.float32)
    unpacked = unpacked.view(unpacked.shape[0], -1, block_size) * scale.unsqueeze(-1)
    return unpacked.reshape(unpacked.shape[0], -1).to(dtype)


def load_nvfp4_model(model_path: str, original_model_path: str):
    """Load NVFP4 quantized model using QDQ modules for dequantization."""
    print(f"Loading quantized model from {model_path}...")

    # Load all quantized tensors
    shard_files = sorted(glob.glob(f"{model_path}/model-*.safetensors"))
    all_tensors = {}
    for f in shard_files:
        with safe_open(f, framework="pt") as sf:
            for key in sf.keys():
                all_tensors[key] = sf.get_tensor(key)

    # Load original model in bf16
    model = AutoModelForImageTextToText.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Replace expert weights with dequantized versions
    replaced = 0
    expert_keys = set()
    for key in all_tensors:
        if ".mlp.experts." in key and key.endswith(".weight") and "scale" not in key:
            expert_keys.add(key)

    for weight_key in sorted(expert_keys):
        scale_key = weight_key.replace(".weight", ".weight_scale")
        scale2_key = weight_key.replace(".weight", ".weight_scale_2")
        if scale_key not in all_tensors or scale2_key not in all_tensors:
            continue

        packed_w = all_tensors[weight_key]  # uint8
        scale = all_tensors[scale_key]  # float8_e4m3fn
        scale2 = all_tensors[scale2_key]  # scalar

        dq_weight = nvfp4_dequantize(packed_w, scale, scale2)

        # Navigate to the parameter in the model
        # key format: model.language_model.layers.X.mlp.experts.Y.{gate,up,down}_proj.weight
        # But original model uses fused gate_up_proj[expert_idx] format
        # We need to write back to the fused tensor
        parts = weight_key.split(".")
        # Find expert idx and proj type
        expert_idx = None
        proj_type = None
        for i, p in enumerate(parts):
            if p == "experts" and i + 1 < len(parts):
                try:
                    expert_idx = int(parts[i + 1])
                except ValueError:
                    pass
            if p in ("gate_proj", "up_proj", "down_proj"):
                proj_type = p
                break

        if expert_idx is None or proj_type is None:
            continue

        # Get layer index
        layer_idx = int(parts[parts.index("layers") + 1])
        layer = model.model.language_model.layers[layer_idx]
        experts = layer.mlp.experts

        if proj_type == "down_proj":
            experts.down_proj.data[expert_idx] = dq_weight.to(torch.bfloat16)
        elif proj_type == "gate_proj":
            # gate_up_proj[expert_idx] = cat(gate, up) along dim 0
            intermediate = dq_weight.shape[0]
            experts.gate_up_proj.data[expert_idx, :intermediate] = dq_weight.to(torch.bfloat16)
        elif proj_type == "up_proj":
            intermediate = dq_weight.shape[0]
            experts.gate_up_proj.data[expert_idx, intermediate:] = dq_weight.to(torch.bfloat16)

        replaced += 1

    print(f"Dequantized and replaced {replaced} expert weight tensors")
    model = model.to("cuda")
    model.eval()
    return model


def eval_ppl(model, tokenizer, seqlen=2048):
    """Evaluate WikiText-2 PPL."""
    from datasets import load_dataset

    wikitext_path = (
        "/root/.cache/huggingface/hub/datasets--Salesforce--wikitext/"
        "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1"
    )
    ds = load_dataset(wikitext_path, split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    max_samples = input_ids.numel() // seqlen

    nlls = []
    with torch.no_grad():
        for i in tqdm(range(max_samples), desc="PPL eval"):
            batch = input_ids[:, i * seqlen : (i + 1) * seqlen].to(model.device)
            outputs = model(input_ids=batch, labels=batch)
            nlls.append(outputs.loss.float().item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


if __name__ == "__main__":
    ORIGINAL = (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
        "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
    )

    models = {
        "standard": "./output/qwen3_5_a3b_nvfp4_gptq/qwen3_5-a3b_nvfp4_gptq",
        "4/6": "./output/qwen3_6_a3b_nvfp4_gptq_46/qwen3_5-a3b_nvfp4_gptq_46",
    }

    available = {k: v for k, v in models.items() if Path(v).exists()}
    if not available:
        print("No quantized models found.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)

    results = {}
    for name, path in available.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        model = load_nvfp4_model(path, ORIGINAL)
        ppl = eval_ppl(model, tokenizer)
        results[name] = ppl
        print(f"  {name}: PPL = {ppl:.4f}")
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    bf16_ppl = 6.3356
    for name, ppl in results.items():
        delta = (ppl - bf16_ppl) / bf16_ppl * 100
        print(f"  {name:12s}: PPL = {ppl:.4f}  (+{delta:.3f}%)")
    if "standard" in results and "4/6" in results:
        std_loss = results["standard"] - bf16_ppl
        fo_loss = results["4/6"] - bf16_ppl
        reduction = (std_loss - fo_loss) / std_loss * 100 if std_loss > 0 else 0
        print(f"\n  BF16 baseline: PPL = {bf16_ppl:.4f}")
        print(f"  4/6 reduces quantization loss by {reduction:.1f}%")
