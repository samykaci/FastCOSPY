"""
Integration tests for UNetLSTMAttention.

All tests use randomly-initialised weights and run entirely offline.
They verify:
  - Model construction with default and custom hyper-parameters
  - Output tensor shapes for a variety of inputs
  - Numerical properties (finiteness, determinism, batch independence)
  - Gradient flow for back-propagation
  - Expected failure modes (wrong dtype, wrong channel count, input too small)

Architecture recap (from models.py)
-------------------------------------
Channel sizes: C1=96, C2=192, C3=384, C4=768, C5=1536, latent=128
Encoder depth: 4 × Down (MaxPool2d + DoubleConvolution)
Decoder depth: 4 × Up   (Upsample  + DoubleConvolution)
Bottleneck:    SpatialAttentionGate(C4, C4, C3)
Energy head:   per-pixel LSTM(latent → lstm_hidden) → Linear(1) × n_energies

Minimum safe spatial size: ≥ 16 in each dimension so that 4 successive
MaxPool2d(2,2) operations never produce a zero-size feature map.
"""

import pytest
import torch
import torch.nn as nn

from fastcospy import SpatialAttentionGate, UNetLSTMAttention


# ---------------------------------------------------------------------------
# Module-scoped fixture (random weights, no download)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> UNetLSTMAttention:
    m = UNetLSTMAttention(n_channels=14)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:

    def test_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_default_n_energies_is_7(self, model):
        assert model.n_energies == 7

    @pytest.mark.parametrize("n_energies", [1, 3, 5, 7, 10])
    def test_custom_n_energies_stored(self, n_energies):
        m = UNetLSTMAttention(n_channels=14, n_energies=n_energies)
        assert m.n_energies == n_energies

    def test_expected_encoder_submodules_exist(self, model):
        for attr in ("inc", "down1", "down2", "down3", "down4"):
            assert hasattr(model, attr), f"Missing encoder submodule: {attr!r}"

    def test_expected_decoder_submodules_exist(self, model):
        for attr in ("up1", "up2", "up3", "up4"):
            assert hasattr(model, attr), f"Missing decoder submodule: {attr!r}"

    def test_lstm_submodule_exists(self, model):
        assert hasattr(model, "lstm")
        assert isinstance(model.lstm, nn.LSTM)

    def test_energy_head_submodule_exists(self, model):
        assert hasattr(model, "energy_head")
        assert isinstance(model.energy_head, nn.Linear)

    def test_energy_head_output_dim_is_1(self, model):
        assert model.energy_head.out_features == 1

    def test_attention_gate_submodule_exists(self, model):
        assert hasattr(model, "att_bottleneck")
        assert isinstance(model.att_bottleneck, SpatialAttentionGate)

    def test_lstm_input_size(self, model):
        """LSTM input size should equal latent_channels (default 128)."""
        assert model.lstm.input_size == 128

    def test_lstm_hidden_size(self, model):
        assert model.lstm.hidden_size == 128

    def test_lstm_num_layers(self, model):
        assert model.lstm.num_layers == 1

    def test_parameter_count_exceeds_10m(self, model):
        """
        This is a large surrogate model. Guard against accidental pruning
        that would silently reduce capacity.
        """
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 10_000_000, (
            f"Expected > 10M parameters, got {n_params:,}"
        )

    def test_no_nan_or_inf_in_initial_weights(self, model):
        for name, p in model.named_parameters():
            assert torch.all(torch.isfinite(p)), (
                f"Non-finite initial weights in {name!r}"
            )

    def test_custom_latent_channels(self):
        m = UNetLSTMAttention(n_channels=14, latent_channels=64)
        m.eval()
        x = torch.randn(1, 14, 21, 41)
        out = m(x)
        assert out.shape == (1, 7, 21, 41)

    def test_custom_lstm_hidden_size(self):
        m = UNetLSTMAttention(n_channels=14, lstm_hidden_size=64)
        m.eval()
        x = torch.randn(1, 14, 21, 41)
        out = m(x)
        assert out.shape == (1, 7, 21, 41)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


class TestOutputShapes:

    def test_standard_training_patch(self, model):
        x = torch.randn(1, 14, 21, 41)
        assert model(x).shape == (1, 7, 21, 41)

    def test_output_has_4_dimensions(self, model):
        x = torch.randn(1, 14, 21, 41)
        assert model(x).ndim == 4

    def test_output_batch_dim_matches_input(self, model):
        for B in [1, 2, 4]:
            x = torch.randn(B, 14, 21, 41)
            assert model(x).shape[0] == B, f"Batch dim mismatch for B={B}"

    def test_output_energy_dim_is_7(self, model):
        x = torch.randn(1, 14, 21, 41)
        assert model(x).shape[1] == 7

    def test_output_spatial_dims_match_input(self, model):
        x = torch.randn(1, 14, 21, 41)
        out = model(x)
        assert out.shape[2] == x.shape[2]
        assert out.shape[3] == x.shape[3]

    @pytest.mark.parametrize("n_energies", [1, 3, 7])
    def test_custom_n_energies_in_output(self, n_energies):
        m = UNetLSTMAttention(n_channels=14, n_energies=n_energies)
        m.eval()
        out = m(torch.randn(1, 14, 21, 41))
        assert out.shape[1] == n_energies

    @pytest.mark.parametrize("h, w", [(16, 16), (21, 41), (32, 64), (16, 32)])
    def test_various_valid_spatial_sizes(self, h, w):
        """
        Spatial dims ≥ 16 are guaranteed to survive 4 MaxPool2d(2,2) layers.
        The U-Net's skip-connection padding restores the exact original size.
        """
        m = UNetLSTMAttention(n_channels=14)
        m.eval()
        x = torch.randn(1, 14, h, w)
        out = m(x)
        assert out.shape[2] == h
        assert out.shape[3] == w

    def test_single_channel_input(self):
        m = UNetLSTMAttention(n_channels=1)
        m.eval()
        out = m(torch.randn(1, 1, 21, 41))
        assert out.shape == (1, 7, 21, 41)


# ---------------------------------------------------------------------------
# Numerical properties
# ---------------------------------------------------------------------------


class TestNumericalProperties:

    def test_output_is_finite_on_random_input(self, model):
        x = torch.randn(1, 14, 21, 41)
        out = model(x)
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"

    def test_output_is_finite_on_zero_input(self, model):
        out = model(torch.zeros(1, 14, 21, 41))
        assert torch.all(torch.isfinite(out))

    def test_output_is_finite_on_constant_input(self, model):
        out = model(torch.ones(1, 14, 21, 41))
        assert torch.all(torch.isfinite(out))

    def test_deterministic_in_eval_mode(self, model):
        """Repeated forward passes with the same input must give identical output."""
        torch.manual_seed(42)
        x = torch.randn(1, 14, 21, 41)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_stochastic_in_train_mode(self):
        """
        down4 uses Dropout2d(p=0.2). In training mode two forward passes over
        the same input should yield different results with very high probability.
        """
        m = UNetLSTMAttention(n_channels=14)
        m.train()
        x = torch.randn(1, 14, 21, 41)
        out1 = m(x)
        out2 = m(x)
        assert not torch.allclose(out1, out2), (
            "Expected stochastic outputs in train mode (Dropout2d)"
        )

    def test_no_gradient_on_output_in_no_grad_context(self, model):
        x = torch.randn(1, 14, 21, 41)
        with torch.no_grad():
            out = model(x)
        assert not out.requires_grad

    def test_output_requires_grad_in_train_mode(self):
        m = UNetLSTMAttention(n_channels=14)
        m.train()
        x = torch.randn(1, 14, 21, 41)
        out = m(x)
        assert out.requires_grad

    def test_gradient_flows_to_input(self):
        m = UNetLSTMAttention(n_channels=14)
        m.train()
        x = torch.randn(1, 14, 21, 41, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_batch_consistency(self, model):
        """
        GroupNorm normalises within each sample independently, so the
        prediction for sample i should not depend on what else is in the batch.
        """
        torch.manual_seed(99)
        x0 = torch.randn(1, 14, 21, 41)
        x1 = torch.randn(1, 14, 21, 41)
        batch = torch.cat([x0, x1], dim=0)

        out_batch = model(batch)
        out0 = model(x0)
        out1 = model(x1)

        torch.testing.assert_close(out_batch[0], out0[0])
        torch.testing.assert_close(out_batch[1], out1[0])

    def test_different_inputs_give_different_outputs(self, model):
        torch.manual_seed(7)
        x1 = torch.randn(1, 14, 21, 41)
        x2 = torch.randn(1, 14, 21, 41)
        assert not torch.allclose(model(x1), model(x2))

    def test_output_dtype_is_float32(self, model):
        out = model(torch.randn(1, 14, 21, 41))
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Edge and error cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_wrong_input_channel_count_raises(self, model):
        """First Conv2d expects exactly n_channels=14 input channels."""
        x = torch.randn(1, 7, 21, 41)
        with pytest.raises(RuntimeError):
            model(x)

    def test_float64_input_raises(self, model):
        """Model weights are float32; a float64 input produces a type mismatch."""
        x = torch.randn(1, 14, 21, 41, dtype=torch.float64)
        with pytest.raises(RuntimeError):
            model(x)

    def test_input_too_small_for_pooling_raises(self, model):
        """
        After 4 MaxPool2d(2,2) layers, a spatial dim of 8 shrinks to 0.
            8 → 4 → 2 → 1 → 0  (zero-size feature map → RuntimeError)
        """
        x = torch.randn(1, 14, 8, 8)
        with pytest.raises(RuntimeError):
            model(x)

    def test_nan_input_propagates_to_output(self, model):
        """NaN in the input should propagate through — document this behaviour."""
        x = torch.randn(1, 14, 21, 41)
        x[0, 0, 0, 0] = float("nan")
        out = model(x)
        assert torch.any(torch.isnan(out))

    def test_switching_from_train_to_eval_is_safe(self):
        m = UNetLSTMAttention(n_channels=14)
        m.train()
        m(torch.randn(1, 14, 21, 41))
        m.eval()
        m(torch.randn(1, 14, 21, 41))

    @pytest.mark.slow
    def test_large_batch_runs_without_error(self, model):
        x = torch.randn(16, 14, 21, 41)
        out = model(x)
        assert out.shape[0] == 16
