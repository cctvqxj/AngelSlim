import torch

from angelslim.compressor.quant.modules.gptq.gptq_module import GPTQModule
from angelslim.compressor.quant.modules.helper_layer import (
    NVFP4QDQModule,
    compute_nvfp4_fixed_grid_block_scale,
    compute_nvfp4_fixed_grid_weight_scale_2,
    nvfp4_cast_to_grid,
    nvfp4_dequantize_grid,
    nvfp4_fixed_grid_quant_dequant,
    unpack_nvfp4_fixed_grid_codes,
)


def _sample_weight():
    return torch.tensor(
        [
            [
                -6.0,
                -4.9,
                -4.0,
                -3.0,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.9,
                6.0,
            ]
        ],
        dtype=torch.float32,
    )


def _quantize(weight, grid):
    scale2 = compute_nvfp4_fixed_grid_weight_scale_2(weight.abs().amax(), 256)
    blocks = weight.reshape(-1, 16)
    scale = compute_nvfp4_fixed_grid_block_scale(blocks, scale2, grid)
    dequant = nvfp4_fixed_grid_quant_dequant(blocks, scale, scale2, grid)
    codes = nvfp4_cast_to_grid(blocks / (scale.float() * scale2), grid)
    return scale2, scale, codes, dequant.reshape_as(weight)


def test_fixed_grid_codebooks_and_common_level2_scale():
    weight = _sample_weight()
    scale2_by_grid = {}
    for grid in ("g6", "g4", "gint"):
        scale2, _, code, _ = _quantize(weight, grid)
        scale2_by_grid[grid] = scale2
        values = nvfp4_dequantize_grid(code, grid)
        if grid in ("g6", "g4"):
            allowed = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
            assert torch.isin(values.abs(), allowed).all()
        else:
            assert (values >= -7).all()
            assert (values <= 7).all()
            assert not (values == -8).any()
    torch.testing.assert_close(scale2_by_grid["g6"], scale2_by_grid["g4"])
    torch.testing.assert_close(scale2_by_grid["g6"], scale2_by_grid["gint"])


def test_e4m3_roundtrip_is_used_for_all_fixed_grids():
    weight = _sample_weight()
    for grid in ("g6", "g4", "gint"):
        scale2, scale, _, _ = _quantize(weight, grid)
        assert scale.dtype == torch.float8_e4m3fn
        roundtrip = scale.float().to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(scale.float(), roundtrip)
        assert torch.isfinite(scale.float()).all()
        assert scale.float().abs().max() <= torch.finfo(torch.float8_e4m3fn).max
        assert scale2.item() > 0


def test_fake_quant_matches_checkpoint_pack_decode_for_all_grids():
    weight = _sample_weight()
    for grid in ("g6", "g4", "gint"):
        scale2, scale, _, fake = _quantize(weight, grid)
        module = NVFP4QDQModule(
            weight=torch.nn.Parameter(weight.clone()),
            weight_scale=torch.nn.Parameter(scale.squeeze(-1)),
            weight_scale_2=torch.nn.Parameter(scale2),
            bias=None,
            block_size=16,
            input_scale=None,
            fixed_grid=grid,
        )
        checkpoint_decode = module.dequantize(
            module.weight, 16, module.weight_scale, module.weight_scale_2
        )
        torch.testing.assert_close(fake, checkpoint_decode.float())
        packed_codes = unpack_nvfp4_fixed_grid_codes(module.weight)
        expected_codes = nvfp4_cast_to_grid(
            weight.reshape(-1, 16) / (scale.float() * scale2), grid
        ).reshape_as(weight)
        assert torch.equal(packed_codes, expected_codes)


def test_gptq_inner_uses_requested_fixed_grid_and_no_actorder():
    torch.manual_seed(965)
    base = torch.nn.Linear(16, 8, bias=False, dtype=torch.float32)
    calibration = torch.randn(2, 8, 16)
    outputs = {}
    scales2 = {}
    for grid in ("g6", "g4", "gint"):
        layer = torch.nn.Linear(16, 8, bias=False, dtype=torch.float32)
        layer.weight.data.copy_(base.weight.data)
        quantizer = GPTQModule(
            layer,
            quant_bits=4,
            weight_format="nvfp4",
            block_size=16,
            fixed_grid=grid,
            level2_scale_max=256,
        )
        quantizer.add_batch(calibration, None)
        scale, scale2, permutation = quantizer.fasterquant(
            blocksize=16,
            percdamp=0.01,
            group_size=16,
            actorder=False,
            sym=True,
        )
        assert permutation is None
        assert scale.dtype == torch.float8_e4m3fn
        outputs[grid] = layer.weight.detach().clone()
        scales2[grid] = scale2
    torch.testing.assert_close(scales2["g6"], scales2["g4"])
    torch.testing.assert_close(scales2["g6"], scales2["gint"])
    assert not torch.equal(outputs["g6"], outputs["g4"])
    assert not torch.equal(outputs["g6"], outputs["gint"])
