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

import math
import time

import torch

from .....utils import get_tensor_item, print_info
from ...core import compute_scales_with_zero
from ..helper_layer import (
    compute_mxfp4_block_scale,
    compute_nvfp4_block_scale,
    compute_nvfp4_weight_scale_2,
    encode_mxfp4_scale,
    mxfp4_quant_dequant,
    nvfp4_quant_dequant,
)

__all__ = ["GPTQModule"]


class GPTQModule:
    def __init__(self, layer, quant_bits=4, weight_format="int4", block_size=16):
        """
        GPTQ quantization wrapper for neural network layers.

        Args:
            layer: Full-precision torch.nn.Module to quantize (Linear)
            quant_bits: Quantization bitwidth (2-8 bits, default=4)
            weight_format: "int4" (uniform), "nvfp4" (E2M1 + two-level
                scale), or "mxfp4" (E2M1 + E8M0 block scale).
            block_size: FP4 micro-scaling block size (16 for NVFP4, 32 for MXFP4).
        """
        super(GPTQModule, self).__init__()
        self.layer = layer
        self.dev = self.layer.weight.device
        self.w = layer.weight.data.clone()
        self.rows = self.w.shape[0]
        self.columns = self.w.shape[1]
        self.h = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0
        self.quant_bits = quant_bits
        self.weight_format = weight_format
        self.block_size = block_size
        # Per-tensor (level-2) NVFP4 scale, set at the start of fasterquant.
        self.weight_scale_2 = None

    def add_batch(self, inp, out):
        # Handle 4D input (e.g., Conv2d or multi-head attention internals)
        if len(inp.shape) == 4:
            inp = inp[0, 0, :, :]
        if len(inp.shape) == 3 and inp.shape[0] == 1:
            inp = inp[0]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.float()
        tmp = inp.shape[0]  # number of tokens
        inp = inp.t()
        self.h *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        self.h += inp.matmul(inp.t())

    def compute_quant_params(self, x, bits, sym):
        if self.weight_format == "mxfp4":
            scale = compute_mxfp4_block_scale(x, self.block_size)
            return scale, torch.zeros_like(scale)
        if self.weight_format == "nvfp4":
            # Per-block effective scale (block_scale_e4m3 * weight_scale_2).
            # weight_scale_2 is computed once in fasterquant before this runs.
            eff_scale = compute_nvfp4_block_scale(x, self.weight_scale_2)
            return eff_scale, torch.zeros_like(eff_scale)
        return compute_scales_with_zero(x, bits=bits, sym=sym)

    def quant_dequant(self, x, weight_scale, weight_zero):
        if self.weight_format == "mxfp4":
            return mxfp4_quant_dequant(x, weight_scale, self.block_size)
        if self.weight_format == "nvfp4":
            return nvfp4_quant_dequant(x, weight_scale)
        maxq = torch.tensor(2**self.quant_bits - 1, device=x.device)
        q = torch.clamp(torch.round(x / weight_scale) + weight_zero, 0, maxq)
        return weight_scale * (q - weight_zero)

    @torch.no_grad()
    def rtn_quantize(self, group_size=-1, sym=True):
        """Apply weight-only round-to-nearest quantization without calibration.

        This is used when an expert receives no routed calibration tokens, so
        no Hessian is available for GPTQ error compensation. The returned
        scales and zero points use the same layout as ``fasterquant`` and can
        be consumed by the normal GPTQ packing path.
        """
        if self.weight_format != "int4":
            raise ValueError("rtn_quantize currently supports only int4 weights")

        weight = self.w.float()
        effective_group_size = group_size if group_size != -1 else self.columns
        scales = []
        zeros = []
        quantized_groups = []

        for start in range(0, self.columns, effective_group_size):
            group = weight[:, start : start + effective_group_size]
            scale, zero = self.compute_quant_params(
                group,
                bits=self.quant_bits,
                sym=sym,
            )
            scales.append(scale)
            zeros.append(zero)
            quantized_groups.append(self.quant_dequant(group, scale, zero))

        quantized_weight = torch.cat(quantized_groups, dim=1)
        self.layer.weight.data.copy_(quantized_weight.type_as(self.layer.weight.data))

        scale = torch.cat(scales, dim=1)
        zero = torch.cat(zeros, dim=1)
        self.w = self.w.cpu()
        del self.w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scale, zero, None

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=True,
        sym=True,
    ):
        w_weight = self.w.float()

        tick = time.time()

        hessian = self.h.float()
        if torch.isnan(hessian).any():
            print_info("[error] Hessian contains nan!")
            exit()
        self.h.detach().cpu()
        del self.h
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        w_weight[:, dead] = 0

        # NVFP4: per-tensor level-2 scale, computed once from the (dead-zeroed)
        # weights before any per-block scale. GPTQ compensation does not change
        # the amax magnitude, so this stays valid for the whole layer.
        #
        # If ``self.weight_scale_2`` was already set externally (e.g. a shared
        # gate/up or qkv level-2 scale injected by GPTQ.run so fused-GEMM
        # deployment uses one per-tensor scale across the group), keep it and do
        # NOT recompute from this layer's amax alone.
        if self.weight_format == "nvfp4" and self.weight_scale_2 is None:
            self.weight_scale_2 = compute_nvfp4_weight_scale_2(w_weight.abs().amax())

        scale = []
        zero = []
        now_idx = 1
        static_groups = True
        effective_group_size = group_size if group_size != -1 else self.columns
        input_perm = None

        if actorder:
            input_perm = torch.argsort(torch.diag(hessian), descending=True)
            w_weight = w_weight[:, input_perm]
            hessian = hessian[input_perm][:, input_perm]

        if static_groups:
            for i in range(0, self.columns, effective_group_size):
                weight_scale, weight_zero = self.compute_quant_params(
                    w_weight[:, i : (i + effective_group_size)],
                    bits=self.quant_bits,
                    sym=sym,
                )
                scale.append(weight_scale)
                zero.append(weight_zero)

        losses = torch.zeros_like(w_weight)
        q_weight = torch.zeros_like(w_weight)

        while 1 > percdamp > 0:
            try:
                damp = percdamp * torch.mean(torch.diag(hessian))
                diag = torch.arange(self.columns, device=self.dev)
                hessian[diag, diag] += damp
                hessian = torch.linalg.cholesky(hessian)
                hessian = torch.cholesky_inverse(hessian)
                hessian = torch.linalg.cholesky(hessian, upper=True)
                hinv = hessian
                break
            except torch._C._LinAlgError as e:
                print_info(e)
                print_info(f"Cholesky failed with percdamp={percdamp:.5f}")
                percdamp += 0.01

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            w1 = w_weight[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            losses1 = torch.zeros_like(w1)
            hinv1 = hinv[i1:i2, i1:i2]

            for i in range(count):
                w = w1[:, i]
                d = hinv1[i, i]

                if not static_groups:
                    if (i1 + i) % effective_group_size == 0:
                        weight_scale, weight_zero = self.compute_quant_params(
                            w_weight[:, (i1 + i) : (i1 + i + effective_group_size)],
                            bits=self.quant_bits,
                            sym=sym,
                        )

                    if ((i1 + i) // effective_group_size) - now_idx == -1:
                        scale.append(weight_scale)
                        zero.append(weight_zero)
                        now_idx += 1
                else:
                    weight_scale = scale[(i1 + i) // effective_group_size]
                    weight_zero = zero[(i1 + i) // effective_group_size]

                q = self.quant_dequant(w.unsqueeze(1), weight_scale, weight_zero)
                q = q.flatten()
                q1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d**2

                err = (w - q) / d
                w1[:, i:] -= err.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                err1[:, i] = err

            q_weight[:, i1:i2] = q1
            losses[:, i1:i2] = losses1 / 2

            w_weight[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print_info(f" duration: {(time.time() - tick)}")
        print_info(f" avg loss: {torch.sum(losses).item() / self.nsamples}")

        target_weight = self.layer.weight.data
        if input_perm is not None:
            target_weight = target_weight[:, input_perm]
        norm_loss = torch.norm(
            q_weight.reshape(self.layer.weight.shape).type_as(target_weight) - target_weight
        )

        all_norm_loss = [norm_loss]

        print_info(f" norm loss: {list(map(get_tensor_item, all_norm_loss))}")

        self.layer.weight.data.copy_(
            q_weight.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)
        )

        if scale == []:
            scale = weight_scale
            zero = torch.zeros_like(weight_scale)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        if self.weight_format == "mxfp4":
            encoded_scale = encode_mxfp4_scale(scale).cpu()
            zero = torch.zeros_like(encoded_scale)
            losses = losses.cpu()
            q_weight = q_weight.cpu()
            w_weight = w_weight.cpu()
            hessian = hessian.cpu()
            hinv = hinv.cpu()
            del losses, q_weight, w_weight, hessian, hinv
            self.w = self.w.cpu()
            del self.w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return encoded_scale, zero, input_perm

        if self.weight_format == "nvfp4":
            # ``scale`` currently holds the effective scale (block_e4m3 *
            # weight_scale_2). Recover the stored E4M3 block scale and hand the
            # per-tensor level-2 scale back via the second return value. Move
            # both to CPU so the per-layer scales never pile up on GPU across
            # all transformer layers.
            block_scale_e4m3 = (scale / self.weight_scale_2).to(torch.float8_e4m3fn).cpu()
            weight_scale_2 = self.weight_scale_2.detach().clone().cpu()
            losses = losses.cpu()
            q_weight = q_weight.cpu()
            w_weight = w_weight.cpu()
            hessian = hessian.cpu()
            hinv = hinv.cpu()
            del losses, q_weight, w_weight, hessian, hinv
            self.w = self.w.cpu()
            del self.w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return block_scale_e4m3, weight_scale_2, input_perm

        losses = losses.cpu()
        q_weight = q_weight.cpu()
        w_weight = w_weight.cpu()
        hessian = hessian.cpu()
        hinv = hinv.cpu()
        del losses, q_weight, w_weight, hessian, hinv
        self.w = self.w.cpu()
        del self.w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scale, zero, input_perm

    def free(self):
        self.h = None
        self.w = None
        self.losses = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
