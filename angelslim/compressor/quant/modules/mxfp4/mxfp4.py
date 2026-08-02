# Copyright 2025 Tencent Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import torch

from .....utils import print_info
from ..helper_layer import compute_mxfp4_block_scale, encode_mxfp4_scale

__all__ = ["MXFP4"]


class MXFP4:
    """Data-free MXFP4 weight-only quantization runner."""

    def __init__(self, model):
        self.model = model
        self.block_size = self.model.quant_config.quant_algo_info["block_size"]
        self.weight_only = self.model.quant_config.quant_algo_info.get("weight_only", False)
        if self.block_size != 32:
            raise ValueError(f"MXFP4 block_size must be 32, got {self.block_size}.")

    @torch.no_grad()
    def run(self, dataloader=None):
        if not self.weight_only:
            raise NotImplementedError("MXFP4 currently supports weight-only quantization only.")
        print_info("Use MXFP4 weight-only mode (no calibration needed)")

    @torch.no_grad()
    def post_process(self, sub_layer, name):
        scale = compute_mxfp4_block_scale(sub_layer.weight.detach(), self.block_size)
        self.model.weight_scales_dict[name] = encode_mxfp4_scale(scale)
