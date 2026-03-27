"""
Unit tests for the four neural-network building blocks:
    DoubleConvolution, Down, Up, SpatialAttentionGate.

All tests use randomly-initialised modules — no pretrained weights are
required, so the entire file runs offline.

Architecture notes (from models.py)
------------------------------------
* GroupNorm is used throughout with GN_GROUPS = 8 groups by default.
  Therefore out_channels must be divisible by 8 unless a custom `groups`
  value is passed.
* Down = MaxPool2d(2,2) → DoubleConvolution   (halves spatial dims)
* Up   = Upsample(×2) or ConvTranspose2d → concat skip → DoubleConvolution
  The `x` tensor fed into Up.forward() should have in_channels // 2 channels;
  the skip tensor should also have in_channels // 2 channels so that the
  concatenation produces exactly in_channels channels for DoubleConvolution.
* SpatialAttentionGate gates `x` using `g` via a sigmoid attention map α ∈ [0,1],
  so |output| ≤ |x| element-wise.
"""

import pytest
import torch
import torch.nn as nn

from fastcospy.models import DoubleConvolution, Down, SpatialAttentionGate, Up


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand(b: int = 2, c: int = 16, h: int = 32, w: int = 32) -> torch.Tensor:
    return torch.randn(b, c, h, w)


# ---------------------------------------------------------------------------
# DoubleConvolution
# ---------------------------------------------------------------------------


class TestDoubleConvolution:

    # --- Construction ---

    def test_is_nn_module(self):
        assert isinstance(DoubleConvolution(16, 32), nn.Module)

    def test_no_dropout_by_default(self):
        block = DoubleConvolution(16, 32)
        # Default: 2 × (Conv2d → GroupNorm → ReLU) = 6 layers
        assert len(block.block) == 6

    def test_dropout_appended_when_nonzero(self):
        block = DoubleConvolution(16, 32, dropout=0.3)
        assert len(block.block) == 7
        assert isinstance(block.block[-1], nn.Dropout2d)

    def test_dropout_not_appended_when_zero(self):
        block = DoubleConvolution(16, 32, dropout=0.0)
        assert len(block.block) == 6

    def test_invalid_out_channels_for_groupnorm_raises(self):
        """GroupNorm requires out_channels % groups == 0."""
        with pytest.raises(Exception):
            # 17 is not divisible by 8
            DoubleConvolution(16, 17, groups=8)

    def test_custom_groups_accepted(self):
        # 14 channels, 7 groups → 14 % 7 == 0 ✓
        block = DoubleConvolution(8, 14, groups=7)
        out = block(_rand(c=8, h=16, w=16))
        assert out.shape[1] == 14

    # --- Output shape ---

    def test_output_shape_preserves_spatial_dims(self):
        block = DoubleConvolution(16, 32)
        x = _rand(b=2, c=16, h=24, w=48)
        assert block(x).shape == (2, 32, 24, 48)

    def test_output_channels_correct(self):
        block = DoubleConvolution(8, 64)
        assert block(_rand(c=8)).shape[1] == 64

    @pytest.mark.parametrize("B", [1, 3, 8])
    def test_batch_size_preserved(self, B):
        block = DoubleConvolution(16, 32)
        assert block(torch.randn(B, 16, 20, 20)).shape[0] == B

    # Verify every encoder channel expansion used in the real U-Net
    @pytest.mark.parametrize(
        "in_c, out_c",
        [(14, 96), (96, 192), (192, 384), (384, 768), (768, 768)],
    )
    def test_architecture_channel_pairs(self, in_c, out_c):
        block = DoubleConvolution(in_c, out_c)
        out = block(_rand(c=in_c, h=32, w=32))
        assert out.shape[1] == out_c

    # --- Numerical properties ---

    def test_output_is_finite(self):
        block = DoubleConvolution(16, 32)
        assert torch.all(torch.isfinite(block(_rand(c=16))))

    def test_gradient_flows_to_input(self):
        block = DoubleConvolution(16, 32)
        x = _rand(c=16).requires_grad_(True)
        block(x).sum().backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_has_trainable_parameters(self):
        block = DoubleConvolution(16, 32)
        n_params = sum(p.numel() for p in block.parameters())
        assert n_params > 0

    def test_eval_and_train_modes_do_not_raise(self):
        block = DoubleConvolution(16, 32, dropout=0.2)
        x = _rand(c=16)
        block.train()
        block(x)
        block.eval()
        block(x)


# ---------------------------------------------------------------------------
# Down
# ---------------------------------------------------------------------------


class TestDown:

    # --- Construction ---

    def test_is_nn_module(self):
        assert isinstance(Down(16, 32), nn.Module)

    def test_has_pool_and_double_conv(self):
        block = Down(16, 32)
        assert isinstance(block.pool, nn.MaxPool2d)
        assert isinstance(block.my_double, DoubleConvolution)

    def test_dropout_propagated_to_inner_block(self):
        block = Down(16, 32, dropout=0.5)
        assert isinstance(block.my_double.block[-1], nn.Dropout2d)

    # --- Output shape ---

    def test_even_spatial_dims_halved(self):
        block = Down(16, 32)
        out = block(_rand(c=16, h=32, w=40))
        assert out.shape == (2, 32, 16, 20)

    def test_odd_spatial_dims_floor_halved(self):
        """MaxPool2d with stride 2 floors for odd input sizes."""
        block = Down(16, 32)
        out = block(_rand(c=16, h=21, w=41))
        assert out.shape == (2, 32, 10, 20)

    def test_output_channels_correct(self):
        block = Down(32, 64)
        assert block(_rand(c=32, h=32, w=32)).shape[1] == 64

    @pytest.mark.parametrize("B", [1, 4])
    def test_batch_size_preserved(self, B):
        block = Down(16, 32)
        assert block(torch.randn(B, 16, 24, 24)).shape[0] == B

    # Verify each encoder step in the actual model
    @pytest.mark.parametrize(
        "in_c, out_c, h, w",
        [
            (96,  192, 21, 41),   # down1: (B,96,21,41)  → (B,192,10,20)
            (192, 384, 10, 20),   # down2: (B,192,10,20) → (B,384,5,10)
            (384, 768,  5, 10),   # down3: (B,384,5,10)  → (B,768,2,5)
            (768, 768,  2,  5),   # down4: (B,768,2,5)   → (B,768,1,2)
        ],
    )
    def test_encoder_step(self, in_c, out_c, h, w):
        block = Down(in_c, out_c)
        out = block(torch.randn(1, in_c, h, w))
        assert out.shape[1] == out_c
        assert out.shape[2] == h // 2
        assert out.shape[3] == w // 2

    # --- Numerical properties ---

    def test_output_is_finite(self):
        block = Down(16, 32)
        assert torch.all(torch.isfinite(block(_rand(c=16))))

    def test_gradient_flows_to_input(self):
        block = Down(16, 32)
        x = _rand(c=16).requires_grad_(True)
        block(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Up
# ---------------------------------------------------------------------------


class TestUp:

    # --- Helpers ---

    @staticmethod
    def _make_x_skip(b=2, c_half=32, h=16, w=16):
        """
        For Up(in_channels=2*c_half, ...), the decoder input x has c_half
        channels and the skip connection also has c_half channels.
        """
        x = torch.randn(b, c_half, h, w)
        skip = torch.randn(b, c_half, h * 2, w * 2)
        return x, skip

    # --- Construction ---

    def test_is_nn_module(self):
        assert isinstance(Up(64, 32), nn.Module)

    def test_bilinear_uses_upsample(self):
        block = Up(64, 32, bilinear=True)
        assert isinstance(block.up, nn.Upsample)

    def test_non_bilinear_uses_conv_transpose(self):
        block = Up(64, 32, bilinear=False)
        assert isinstance(block.up, nn.ConvTranspose2d)

    # --- Output shape (bilinear) ---

    def test_bilinear_output_channels(self):
        block = Up(64, 32, bilinear=True)
        x, skip = self._make_x_skip()
        assert block(x, skip).shape[1] == 32

    def test_bilinear_output_spatial_matches_skip(self):
        block = Up(64, 32, bilinear=True)
        x, skip = self._make_x_skip(h=8, w=8)
        out = block(x, skip)
        assert out.shape[2] == skip.shape[2]
        assert out.shape[3] == skip.shape[3]

    @pytest.mark.parametrize("B", [1, 4])
    def test_bilinear_batch_size_preserved(self, B):
        block = Up(64, 32, bilinear=True)
        x = torch.randn(B, 32, 8, 8)
        skip = torch.randn(B, 32, 16, 16)
        assert block(x, skip).shape[0] == B

    # --- Output shape (transposed conv) ---

    def test_transposed_output_channels(self):
        block = Up(64, 32, bilinear=False)
        x, skip = self._make_x_skip()
        assert block(x, skip).shape[1] == 32

    def test_transposed_output_spatial_matches_skip(self):
        block = Up(64, 32, bilinear=False)
        x, skip = self._make_x_skip(h=8, w=8)
        out = block(x, skip)
        assert out.shape[2] == skip.shape[2]
        assert out.shape[3] == skip.shape[3]

    # --- Odd-dimension padding (the critical U-Net edge case) ---

    def test_skip_one_pixel_taller_is_padded_correctly(self):
        """
        After 4 MaxPool2d(2,2) layers on H=21: 21→10→5→2→1.
        On the way back up: upsample 1→2, but skip has 2 (ok); upsample 2→4,
        skip has 5 (off-by-one). Up must pad x to match skip exactly.
        """
        block = Up(192, 96, bilinear=True)
        x = torch.randn(1, 96, 4, 8)
        skip = torch.randn(1, 96, 5, 16)   # skip is 1 row taller
        out = block(x, skip)
        assert out.shape[2] == 5
        assert out.shape[3] == 16

    def test_skip_one_pixel_wider_is_padded_correctly(self):
        block = Up(192, 96, bilinear=True)
        x = torch.randn(1, 96, 10, 20)
        skip = torch.randn(1, 96, 20, 41)  # skip is 1 col wider (from W=41)
        out = block(x, skip)
        assert out.shape[2] == 20
        assert out.shape[3] == 41

    # Verify each decoder step in the actual model
    @pytest.mark.parametrize(
        "in_c, out_c, x_h, x_w, skip_h, skip_w",
        [
            (1536, 384,  1,  2,  2,  5),   # up1
            ( 768, 192,  2,  5,  5, 10),   # up2
            ( 384,  96,  5, 10, 10, 20),   # up3
            ( 192, 128, 10, 20, 21, 41),   # up4 (odd skip size)
        ],
    )
    def test_decoder_step(self, in_c, out_c, x_h, x_w, skip_h, skip_w):
        block = Up(in_c, out_c, bilinear=True)
        x = torch.randn(1, in_c // 2, x_h, x_w)
        skip = torch.randn(1, in_c // 2, skip_h, skip_w)
        out = block(x, skip)
        assert out.shape[1] == out_c
        assert out.shape[2] == skip_h
        assert out.shape[3] == skip_w

    # --- Numerical properties ---

    def test_bilinear_output_is_finite(self):
        block = Up(64, 32, bilinear=True)
        x, skip = self._make_x_skip()
        assert torch.all(torch.isfinite(block(x, skip)))

    def test_transposed_output_is_finite(self):
        block = Up(64, 32, bilinear=False)
        x, skip = self._make_x_skip()
        assert torch.all(torch.isfinite(block(x, skip)))

    def test_gradient_flows_through_bilinear(self):
        block = Up(64, 32, bilinear=True)
        x, skip = self._make_x_skip()
        x = x.requires_grad_(True)
        block(x, skip).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# SpatialAttentionGate
# ---------------------------------------------------------------------------


class TestSpatialAttentionGate:

    @pytest.fixture
    def gate(self):
        return SpatialAttentionGate(F_g=64, F_l=64, F_int=16)

    @staticmethod
    def _make_gx(b=2, fg=64, fl=64, h=16, w=16):
        g = torch.randn(b, fg, h, w)
        x = torch.randn(b, fl, h, w)
        return g, x

    # --- Construction ---

    def test_is_nn_module(self, gate):
        assert isinstance(gate, nn.Module)

    def test_has_expected_sub_layers(self, gate):
        assert hasattr(gate, "W_g")
        assert hasattr(gate, "W_x")
        assert hasattr(gate, "psi")
        assert hasattr(gate, "relu")

    def test_psi_ends_with_sigmoid(self, gate):
        # psi = Sequential(Conv2d(F_int,1,1), Sigmoid())
        assert isinstance(gate.psi[-1], nn.Sigmoid)

    # --- Output shape ---

    def test_output_shape_matches_x(self, gate):
        g, x = self._make_gx()
        assert gate(x, g).shape == x.shape

    def test_output_channels_equal_fl(self, gate):
        g, x = self._make_gx(fl=64)
        assert gate(x, g).shape[1] == 64

    @pytest.mark.parametrize("B", [1, 3, 8])
    def test_batch_size_preserved(self, B, gate):
        x = torch.randn(B, 64, 16, 16)
        g = torch.randn(B, 64, 16, 16)
        assert gate(x, g).shape[0] == B

    # --- Attention semantics ---

    def test_attention_weight_bounds_output(self):
        """
        The gate computes out = x * α where α = sigmoid(…) ∈ [0, 1].
        Therefore |out| ≤ |x| element-wise.
        """
        gate = SpatialAttentionGate(F_g=64, F_l=64, F_int=16)
        gate.eval()
        g, x = self._make_gx()
        out = gate(x, g)
        assert torch.all(out.abs() <= x.abs() + 1e-5)

    def test_zero_gating_signal_suppresses_output(self):
        """
        With g ≈ 0, W_g(g) ≈ 0, so psi ≈ ReLU(W_x(x)) and α is near
        some baseline — the output should still be finite.
        """
        gate = SpatialAttentionGate(F_g=64, F_l=64, F_int=16)
        gate.eval()
        x = torch.randn(1, 64, 16, 16)
        g = torch.zeros(1, 64, 16, 16)
        out = gate(x, g)
        assert torch.all(torch.isfinite(out))

    # --- Actual architecture dimensions ---

    def test_works_with_bottleneck_dims(self):
        """C4=768, F_int=C3=384 as used inside UNetLSTMAttention."""
        gate = SpatialAttentionGate(F_g=768, F_l=768, F_int=384)
        x = torch.randn(1, 768, 2, 3)
        g = torch.randn(1, 768, 2, 3)
        out = gate(x, g)
        assert out.shape == (1, 768, 2, 3)

    def test_gating_signal_smaller_spatial_dims(self):
        """
        g can have different (smaller) spatial dims — the gate upsamples it
        to match x via F.interpolate before adding.
        """
        gate = SpatialAttentionGate(F_g=32, F_l=32, F_int=16)
        x = torch.randn(1, 32, 16, 16)
        g = torch.randn(1, 32, 8, 8)
        out = gate(x, g)
        assert out.shape == x.shape

    # --- Numerical properties ---

    def test_output_is_finite(self, gate):
        g, x = self._make_gx()
        assert torch.all(torch.isfinite(gate(x, g)))

    def test_gradient_flows_through(self, gate):
        g, x = self._make_gx()
        x = x.requires_grad_(True)
        gate(x, g).sum().backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_deterministic_in_eval_mode(self, gate):
        gate.eval()
        g, x = self._make_gx()
        out1 = gate(x, g)
        out2 = gate(x, g)
        torch.testing.assert_close(out1, out2)
