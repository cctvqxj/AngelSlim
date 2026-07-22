"""PPL comparison for standard NVFP4-AWQ and NVFP4-AWQ + 4/6."""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.ppl_eval_compare import eval_ppl, load_nvfp4_model  # noqa: E402

ORIGINAL = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
MODELS = {
    "standard_awq": ("./output/qwen3_6_a3b_nvfp4_awq/" "qwen3_6-a3b_nvfp4_awq"),
    "awq_4/6": ("./output/qwen3_6_a3b_nvfp4_awq_46/" "qwen3_6-a3b_nvfp4_awq_46"),
}
BF16_PPL = 6.3356


def main():
    missing = [path for path in MODELS.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing quantized checkpoints: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL, trust_remote_code=True)
    results = {}
    for name, path in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 60}")
        model = load_nvfp4_model(path, ORIGINAL)
        results[name] = eval_ppl(model, tokenizer)
        print(f"{name}: PPL = {results[name]:.4f}")
        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("NVFP4-AWQ PPL RESULTS")
    print(f"{'=' * 60}")
    for name, ppl in results.items():
        delta = (ppl - BF16_PPL) / BF16_PPL * 100
        print(f"{name:16s}: PPL = {ppl:.4f}  (+{delta:.3f}%)")

    standard_loss = results["standard_awq"] - BF16_PPL
    four_over_six_loss = results["awq_4/6"] - BF16_PPL
    reduction = (
        (standard_loss - four_over_six_loss) / standard_loss * 100 if standard_loss > 0 else 0.0
    )
    print(f"BF16 baseline: PPL = {BF16_PPL:.4f}")
    print(f"4/6 reduces AWQ quantization loss by {reduction:.1f}%")


if __name__ == "__main__":
    main()
