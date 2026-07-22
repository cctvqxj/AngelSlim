"""PPL comparison for the ShareGPT-128 NVFP4 experiment matrix."""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tools.ppl_eval_compare import eval_ppl, load_nvfp4_model

sys.path.insert(0, str(Path(__file__).parent.parent))


ORIGINAL = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
MODELS = {
    "nvfp4_weight": ("./output/qwen3_6_a3b_nvfp4_weight_only/" "qwen3_6-a3b_nvfp4_weight_only"),
    "gptq": ("./output/qwen3_6_a3b_nvfp4_gptq_sharegpt128/" "qwen3_6-a3b_nvfp4_gptq_sharegpt128"),
    "gptq_4/6": (
        "./output/qwen3_6_a3b_nvfp4_gptq_46_sharegpt128/" "qwen3_6-a3b_nvfp4_gptq_46_sharegpt128"
    ),
    "awq": ("./output/qwen3_6_a3b_nvfp4_awq_sharegpt128/" "qwen3_6-a3b_nvfp4_awq_sharegpt128"),
    "awq_4/6": (
        "./output/qwen3_6_a3b_nvfp4_awq_46_sharegpt128/" "qwen3_6-a3b_nvfp4_awq_46_sharegpt128"
    ),
}
BF16_PPL = 6.3356


def main():
    available = {name: path for name, path in MODELS.items() if Path(path).exists()}
    missing = sorted(set(MODELS) - set(available))
    if missing:
        print(f"Skipping unavailable checkpoints: {missing}")
    if not available:
        raise FileNotFoundError("No ShareGPT-128 NVFP4 checkpoints found.")

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)
    results = {}
    for name, path in available.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 60}")
        model = load_nvfp4_model(path, ORIGINAL)
        results[name] = eval_ppl(model, tokenizer)
        print(f"{name}: PPL = {results[name]:.4f}")
        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("SHAREGPT-128 NVFP4 PPL RESULTS")
    print(f"{'=' * 60}")
    for name, ppl in results.items():
        delta = (ppl - BF16_PPL) / BF16_PPL * 100
        print(f"{name:16s}: PPL = {ppl:.4f}  ({delta:+.3f}%)")


if __name__ == "__main__":
    main()
