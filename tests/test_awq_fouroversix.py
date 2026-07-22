import torch

from angelslim.compressor.quant.core import pseudo_quantize_tensor
from angelslim.compressor.quant.modules.awq.search import AWQSearch
from angelslim.compressor.quant.modules.helper_layer import (
    compute_nvfp4_block_scale_fouroversix,
    compute_nvfp4_weight_scale_2_fouroversix,
    nvfp4_quant_dequant,
    nvfp4_quant_dequant_fouroversix,
)
from angelslim.compressor.quant.modules.nvfp4.nvfp4 import NVFP4


def test_pseudo_quantize_tensor_routes_nvfp4_fouroversix():
    weight = torch.tensor(
        [
            [
                6.0,
                4.1,
                3.9,
                3.0,
                2.1,
                1.9,
                1.5,
                1.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -4.2,
                -6.0,
            ]
        ],
        dtype=torch.float32,
    )

    actual = pseudo_quantize_tensor(
        weight,
        weight_format="nvfp4",
        four_over_six=True,
        block_size=16,
    )

    blocks = weight.reshape(-1, 16)
    scale_2 = compute_nvfp4_weight_scale_2_fouroversix(blocks.abs().amax())
    eff_6, eff_4 = compute_nvfp4_block_scale_fouroversix(blocks, scale_2)
    expected = nvfp4_quant_dequant_fouroversix(blocks, eff_6, eff_4).reshape_as(weight)

    torch.testing.assert_close(actual, expected)


def test_pseudo_quantize_tensor_keeps_int4_default_behavior():
    weight = torch.tensor([[-1.0, -0.2, 0.3, 1.0]], dtype=torch.float32)
    default = pseudo_quantize_tensor(weight, w_bit=4, q_group_size=4)
    explicit = pseudo_quantize_tensor(weight, w_bit=4, q_group_size=4, weight_format="int4")
    torch.testing.assert_close(default, explicit)


def test_awq_search_carries_nvfp4_fouroversix_options():
    search = AWQSearch(
        weight_format="nvfp4",
        four_over_six=True,
        block_size=16,
    )
    assert search.weight_format == "nvfp4"
    assert search.four_over_six is True
    assert search.block_size == 16


def test_nvfp4_export_scale_uses_same_fouroversix_choice():
    runner = NVFP4.__new__(NVFP4)
    runner.four_over_six = True
    weight = torch.tensor(
        [
            [
                6.0,
                4.1,
                3.9,
                3.0,
                2.1,
                1.9,
                1.5,
                1.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -4.2,
                -6.0,
            ]
        ],
        dtype=torch.float32,
    )

    scale_2 = runner.get_weights_scaling_factor_2(weight.abs().amax())
    stored_scale = runner.get_weights_scaling_factor(weight, 16, scale_2)
    actual = nvfp4_quant_dequant(
        weight.reshape(-1, 16),
        stored_scale.float().reshape(-1, 1) * scale_2,
    )

    blocks = weight.reshape(-1, 16)
    eff_6, eff_4 = compute_nvfp4_block_scale_fouroversix(blocks, scale_2)
    expected = nvfp4_quant_dequant_fouroversix(blocks, eff_6, eff_4)
    torch.testing.assert_close(actual, expected)
