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
    compute_nvfp4_block_scale,
    compute_nvfp4_block_scale_fouroversix,
    compute_nvfp4_fixed_grid_block_scale,
    compute_nvfp4_fixed_grid_weight_scale_2,
    compute_nvfp4_weight_scale_2,
    compute_nvfp4_weight_scale_2_fouroversix,
    normalize_nvfp4_grid,
    nvfp4_fixed_grid_quant_dequant,
    nvfp4_quant_dequant,
    nvfp4_quant_dequant_fouroversix,
)

__all__ = ["GPTQModule"]


class GPTQModule:
    def __init__(
        self,
        layer,
        quant_bits=4,
        weight_format="int4",
        block_size=16,
        four_over_six=False,
        fixed_grid=None,
        level2_scale_max=256.0,
    ):
        """
        GPTQ quantization wrapper for neural network layers.

        Args:
            layer: Full-precision torch.nn.Module to quantize (Linear)
            quant_bits: Quantization bitwidth (2-8 bits, default=4)
            weight_format: "int4" (default, uniform) or "nvfp4" (E2M1 grid +
                two-level scale). Routes compute_quant_params / quant_dequant.
            block_size: NVFP4 micro-scaling block size (nvfp4 only).
            four_over_six: If True, use adaptive 4/6 block scaling (nvfp4 only).
            fixed_grid: Controlled fixed grid: ``g6``, ``g4``, or ``gint``.
            level2_scale_max: Common level-2 denominator multiplier. The
                controlled experiment fixes this to 256 for all three grids.
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
        self.four_over_six = four_over_six
        self.fixed_grid = normalize_nvfp4_grid(fixed_grid) if fixed_grid is not None else None
        self.level2_scale_max = float(level2_scale_max)
        if self.fixed_grid is not None and self.four_over_six:
            raise ValueError("fixed_grid and adaptive four_over_six are mutually exclusive.")
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
        if self.weight_format == "nvfp4":
            if self.fixed_grid is not None:
                block_scale = compute_nvfp4_fixed_grid_block_scale(
                    x, self.weight_scale_2, self.fixed_grid
                )
                return block_scale, torch.zeros_like(block_scale)
            if self.four_over_six:
                eff_6, eff_4 = compute_nvfp4_block_scale_fouroversix(x, self.weight_scale_2)
                return (eff_6, eff_4), torch.zeros(1, device=x.device)
            eff_scale = compute_nvfp4_block_scale(x, self.weight_scale_2)
            return eff_scale, torch.zeros_like(eff_scale)
        return compute_scales_with_zero(x, bits=bits, sym=sym)

    def quant_dequant(self, x, weight_scale, weight_zero):
        if self.weight_format == "nvfp4":
            if self.fixed_grid is not None:
                return nvfp4_fixed_grid_quant_dequant(
                    x, weight_scale, self.weight_scale_2, self.fixed_grid
                )
            if self.four_over_six:
                eff_6, eff_4 = weight_scale
                return nvfp4_quant_dequant_fouroversix(x, eff_6, eff_4)
            return nvfp4_quant_dequant(x, weight_scale)
        maxq = torch.tensor(2**self.quant_bits - 1, device=x.device)
        q = torch.clamp(torch.round(x / weight_scale) + weight_zero, 0, maxq)
        return weight_scale * (q - weight_zero)

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=True,
        sym=True,
    ):
        if self.nsamples == 0:
            if self.weight_format != "nvfp4":
                print_info(
                    "[warn] nsamples=0, skipping fasterquant (no calibration data routed here)"
                )
                scale = torch.zeros(0)
                zero = torch.zeros(0)
                return scale, zero, None

            # A routed expert may receive no tokens in the fixed 16-sample
            # calibration set. It still must be stored as four-bit weight
            # rather than silently left BF16. With no Hessian information the
            # well-defined limit is RTN using the exact same fixed-grid
            # primitive and shared level-2 scale as the observed experts.
            print_info(
                "[warn] nsamples=0, applying fixed-grid RTN because no "
                "calibration tokens were routed to this expert"
            )
            w_weight = self.w.float()
            if self.weight_scale_2 is None:
                if self.fixed_grid is not None:
                    self.weight_scale_2 = compute_nvfp4_fixed_grid_weight_scale_2(
                        w_weight.abs().amax(), self.level2_scale_max
                    )
                elif self.four_over_six:
                    self.weight_scale_2 = compute_nvfp4_weight_scale_2_fouroversix(
                        w_weight.abs().amax()
                    )
                else:
                    self.weight_scale_2 = compute_nvfp4_weight_scale_2(w_weight.abs().amax())

            if self.fixed_grid is None:
                raise ValueError(
                    "The controlled no-activation fallback is implemented only "
                    "for fixed_grid experiments."
                )
            blocks = w_weight.reshape(self.rows, -1, self.block_size)
            block_scale = compute_nvfp4_fixed_grid_block_scale(
                blocks, self.weight_scale_2, self.fixed_grid
            )
            q_weight = nvfp4_fixed_grid_quant_dequant(
                blocks, block_scale, self.weight_scale_2, self.fixed_grid
            ).reshape_as(w_weight)
            self.layer.weight.data.copy_(q_weight.type_as(self.layer.weight.data))
            self.w = self.w.cpu()
            del self.w
            return (
                block_scale.squeeze(-1).to(torch.float8_e4m3fn).cpu(),
                self.weight_scale_2.detach().clone().cpu(),
                None,
            )

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
            if self.fixed_grid is not None:
                self.weight_scale_2 = compute_nvfp4_fixed_grid_weight_scale_2(
                    w_weight.abs().amax(), self.level2_scale_max
                )
            elif self.four_over_six:
                self.weight_scale_2 = compute_nvfp4_weight_scale_2_fouroversix(
                    w_weight.abs().amax()
                )
            else:
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
        print_info(f" avg loss: {torch.sum(losses).item() / max(self.nsamples, 1)}")

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
            zero = (
                torch.zeros_like(weight_scale)
                if not isinstance(weight_scale, tuple)
                else torch.zeros(1)
            )
        if self.weight_format == "nvfp4" and self.four_over_six:
            pass  # scale entries are tuples; handled below in the 4/6 branch
        elif isinstance(scale, list):
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)

        if self.weight_format == "nvfp4":
            # ``scale`` currently holds the effective scale (block_e4m3 *
            # weight_scale_2). Recover the stored E4M3 block scale and hand the
            # per-tensor level-2 scale back via the second return value. Move
            # both to CPU so the per-layer scales never pile up on GPU across
            # all transformer layers.
            if self.four_over_six:
                # With 4/6, scale entries are tuples (eff_6, eff_4). We need to
                # re-derive the actual block scales from the final compensated
                # weight using 4/6 selection on the final state.
                final_w = self.layer.weight.data.float()
                if input_perm is not None:
                    final_w = final_w[:, input_perm]
                final_blocks = final_w.reshape(self.rows, -1, self.block_size)
                eff_6, eff_4 = compute_nvfp4_block_scale_fouroversix(
                    final_blocks.reshape(-1, self.block_size), self.weight_scale_2
                )
                dq6 = nvfp4_quant_dequant(final_blocks.reshape(-1, self.block_size), eff_6)
                dq4 = nvfp4_quant_dequant(final_blocks.reshape(-1, self.block_size), eff_4)
                orig = final_blocks.reshape(-1, self.block_size)
                err6 = ((dq6 - orig) ** 2).sum(-1, keepdim=True)
                err4 = ((dq4 - orig) ** 2).sum(-1, keepdim=True)
                chosen_eff = torch.where(err4 < err6, eff_4, eff_6)
                chosen_eff = chosen_eff.reshape(self.rows, -1)
                block_scale_e4m3 = (chosen_eff / self.weight_scale_2).to(torch.float8_e4m3fn).cpu()
            else:
                if isinstance(scale, list):
                    scale = torch.cat(scale, dim=1)
                block_scale_e4m3 = (
                    scale.to(torch.float8_e4m3fn).cpu()
                    if self.fixed_grid is not None
                    else (scale / self.weight_scale_2).to(torch.float8_e4m3fn).cpu()
                )
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
