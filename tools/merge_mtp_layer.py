"""Merge MTP layer (model.layers.78.*) from reference model into our weight-only output.

Creates a new output directory with:
- All safetensor shards from our model (symlinked)
- One new shard containing model.layers.78.* tensors from reference
- Updated model.safetensors.index.json
- All other JSON/config files copied from reference
"""

import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

SRC_PATH = (
    "/apdcephfs_hldy2/share_300532381/xiaojingqi/glm/"
    "GLM-5-1-nvfp4-weight-only/glm5_1_nvfp4_weight_only_internal"
)
REF_PATH = (
    "/apdcephfs_hldy2/share_300532381/rubingyang/models/glm/"
    "NVIDIA-GLM-5.1-NVFP4-share_gate_up_w2_fix"
)
OUT_PATH = (
    "/apdcephfs_hldy2/share_300532381/xiaojingqi/glm/"
    "GLM-5-1-nvfp4-weight-only/glm5_1_nvfp4_weight_only_internal_mtp"
)

os.makedirs(OUT_PATH, exist_ok=True)

# 1. Copy JSON files from reference (except model.safetensors.index.json)
for fname in os.listdir(REF_PATH):
    if fname.endswith(".json") and fname != "model.safetensors.index.json":
        shutil.copy2(os.path.join(REF_PATH, fname), os.path.join(OUT_PATH, fname))
        print(f"Copied {fname} from reference")

# Also copy tokenizer files if present
for fname in os.listdir(REF_PATH):
    if "tokenizer" in fname and not fname.endswith(".json"):
        shutil.copy2(os.path.join(REF_PATH, fname), os.path.join(OUT_PATH, fname))
        print(f"Copied {fname} from reference")

# 2. Load our index
with open(os.path.join(SRC_PATH, "model.safetensors.index.json")) as f:
    src_idx = json.load(f)

# 3. Load reference index, find layer 78 keys and their shards
with open(os.path.join(REF_PATH, "model.safetensors.index.json")) as f:
    ref_idx = json.load(f)

layer78_keys = {k: v for k, v in ref_idx["weight_map"].items() if k.startswith("model.layers.78.")}
ref_shards_needed = sorted(set(layer78_keys.values()))
print(f"\nLayer 78: {len(layer78_keys)} tensors across {len(ref_shards_needed)} ref shards")

# 4. Hard-link existing src shards into output (no copy, instant, same filesystem)
for fname in sorted(set(src_idx["weight_map"].values())):
    src_file = os.path.join(SRC_PATH, fname)
    out_file = os.path.join(OUT_PATH, fname)
    if not os.path.exists(out_file):
        os.link(src_file, out_file)

print(f"Hard-linked {len(set(src_idx['weight_map'].values()))} shards from source")

# 5. Extract layer 78 tensors from reference and save as new shards
#    Keep original shard grouping to avoid huge single files
new_weight_map = dict(src_idx["weight_map"])

for ref_shard_name in ref_shards_needed:
    # New shard name: append after our existing shards
    # Use "mtp-" prefix to avoid collision
    new_shard_name = f"mtp-{ref_shard_name}"
    ref_shard_path = os.path.join(REF_PATH, ref_shard_name)
    out_shard_path = os.path.join(OUT_PATH, new_shard_name)

    # Load only layer 78 keys from this shard
    keys_in_shard = [k for k, v in layer78_keys.items() if v == ref_shard_name]
    tensors = {}
    with safe_open(ref_shard_path, framework="pt") as f:
        for k in keys_in_shard:
            tensors[k] = f.get_tensor(k)

    save_file(tensors, out_shard_path)
    print(f"Saved {len(tensors)} tensors to {new_shard_name}")

    for k in keys_in_shard:
        new_weight_map[k] = new_shard_name

# 6. Write new index
new_idx = {
    "metadata": src_idx.get("metadata", {}),
    "weight_map": dict(sorted(new_weight_map.items())),
}
with open(os.path.join(OUT_PATH, "model.safetensors.index.json"), "w") as f:
    json.dump(new_idx, f, indent=2)

print(f"\nDone! Output: {OUT_PATH}")
print(f"Total keys in new index: {len(new_weight_map)}")
