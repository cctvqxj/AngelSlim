"""Convert ShareGPT conversations to deterministic chat-templated calibration data."""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def normalize_conversation(record):
    messages = []
    for turn in record.get("conversations", []):
        role = ROLE_MAP.get(turn.get("from", turn.get("role", "")).lower())
        content = turn.get("value", turn.get("content"))
        if role is None or content is None or not str(content).strip():
            continue
        messages.append({"role": role, "content": str(content)})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            messages = normalize_conversation(record)
            if messages:
                records.append(messages)

    if len(records) < args.num_samples:
        raise ValueError(
            f"Only {len(records)} valid conversations, fewer than {args.num_samples} requested."
        )

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(records)), args.num_samples))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for source_index in selected_indices:
            applied_message = tokenizer.apply_chat_template(
                records[source_index],
                tokenize=False,
                add_generation_prompt=False,
            )
            json.dump(
                {
                    "applied_message": applied_message,
                    "source_index": source_index,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    print(
        f"Wrote {args.num_samples} samples to {output_path} "
        f"(seed={args.seed}, valid_source_records={len(records)})"
    )


if __name__ == "__main__":
    main()
