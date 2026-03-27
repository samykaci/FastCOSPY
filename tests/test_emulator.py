"""
End-to-end tests for FastCOSPYEmulator.

The `emulator` fixture (conftest.py) is session-scoped: pretrained weights are
downloaded once and cached in ~/.fastcospy. After the first run the tests are
fully offline.

Coverage
--------
* Initialization: device, mode, weight file, model structure
* predict() shapes: batch, energy, and spatial dimensions
* predict() numerics: finiteness, determinism, batch independence, value range
* Error handling: wrong channels, wrong dtype, numpy array, spatial dims too small
* Known propagation behaviour: NaN / Inf inputs
"""

import numpy as np
import pytest
import torch

from fastcospy import FastCOSPYEmulator
from fastcospy.schema import ENERGY_CENTER


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:

    def test_emulator_is_not_none(self, emulator):
        assert emulator is not None

    def test_model_attribute_exists(self, emulator):
        assert hasattr(emulator, "model")

    def test_model_in_eval_mode(self, emulator):
        assert not emulator.model.training, (
            "model.training should be False after FastCOSPYEmulator.__init__"
        )

    def test_device_defaults_to_cpu(self, emulator):
        assert emulator.device == "cpu"

    def test_weights_path_attribute_exists(self, emulator):
        assert hasattr(emulator, "weights_path")

    def test_weights_file_exists_on_disk(self, emulator):
        assert emulator.weights_path.exists(), (
            "Cached weights file must exist after initialisation"
        )

    def test_weights_path_is_a_file(self, emulator):
        assert emulator.weights_path.is_file()

    def test_weights_file_is_non_empty(self, emulator):
        assert emulator.weights_path.stat().st_size > 0

    def test_model_accepts_14_channel_input(self, emulator):
        x = torch.randn(1, 14, 21, 41)
        out = emulator.predict(x)
        assert out is not None

    def test_incompatible_n_channels_raises_on_weight_load(self):
        """
        The v1.0.0 weights were trained for n_channels=14. Requesting a
        different value produces a state-dict size mismatch → RuntimeError.
        """
        with pytest.raises(RuntimeError):
            FastCOSPYEmulator(n_channels=7)

    def test_second_instantiation_reuses_cache(self, emulator):
        """
        Creating a second emulator must not fail even though the weights
        file is already cached (download_weights must be idempotent).
        """
        emulator2 = FastCOSPYEmulator(device="cpu")
        assert emulator2.weights_path == emulator.weights_path


# ---------------------------------------------------------------------------
# predict() — output shapes
# ---------------------------------------------------------------------------


class TestPredictShape:

    def test_standard_patch_shape(self, emulator):
        x = torch.randn(1, 14, 21, 41)
        assert emulator.predict(x).shape == (1, 7, 21, 41)

    def test_output_is_4d(self, emulator, standard_input):
        assert emulator.predict(standard_input).ndim == 4

    def test_batch_dim_1(self, emulator, standard_input):
        assert emulator.predict(standard_input).shape[0] == 1

    def test_batch_dim_4(self, emulator, batch_input):
        assert emulator.predict(batch_input).shape[0] == 4

    def test_energy_dim_is_7(self, emulator, standard_input):
        assert emulator.predict(standard_input).shape[1] == 7

    def test_spatial_height_preserved(self, emulator):
        x = torch.randn(1, 14, 21, 41)
        assert emulator.predict(x).shape[2] == 21

    def test_spatial_width_preserved(self, emulator):
        x = torch.randn(1, 14, 21, 41)
        assert emulator.predict(x).shape[3] == 41

    def test_output_is_tensor(self, emulator, standard_input):
        assert isinstance(emulator.predict(standard_input), torch.Tensor)

    def test_output_lives_on_cpu(self, emulator, standard_input):
        assert emulator.predict(standard_input).device.type == "cpu"

    def test_output_dtype_is_float32(self, emulator, standard_input):
        assert emulator.predict(standard_input).dtype == torch.float32

    @pytest.mark.parametrize("B", [1, 2, 4])
    def test_various_batch_sizes(self, emulator, B):
        x = torch.randn(B, 14, 21, 41)
        out = emulator.predict(x)
        assert out.shape[0] == B
        assert out.shape[1] == 7


# ---------------------------------------------------------------------------
# predict() — numerical properties
# ---------------------------------------------------------------------------


class TestPredictNumerics:

    def test_output_is_finite(self, emulator, standard_input):
        out = emulator.predict(standard_input)
        assert torch.all(torch.isfinite(out)), "Prediction must not contain NaN or Inf"

    def test_output_has_no_gradient(self, emulator, standard_input):
        """predict() runs inside torch.no_grad() — output must not carry grad."""
        out = emulator.predict(standard_input)
        assert not out.requires_grad

    def test_deterministic_across_calls(self, emulator, standard_input):
        """Two consecutive calls with the same input must produce identical output."""
        out1 = emulator.predict(standard_input)
        out2 = emulator.predict(standard_input)
        torch.testing.assert_close(out1, out2)

    def test_batch_consistency(self, emulator):
        """
        GroupNorm normalises per-sample, so predict([a, b]) must equal
        [predict(a), predict(b)] element-wise.
        """
        torch.manual_seed(0)
        x0 = torch.randn(1, 14, 21, 41)
        x1 = torch.randn(1, 14, 21, 41)
        batch = torch.cat([x0, x1], dim=0)

        out_batch = emulator.predict(batch)
        out0 = emulator.predict(x0)
        out1 = emulator.predict(x1)

        torch.testing.assert_close(out_batch[0], out0[0])
        torch.testing.assert_close(out_batch[1], out1[0])

    def test_different_inputs_give_different_outputs(self, emulator):
        torch.manual_seed(42)
        x1 = torch.randn(1, 14, 21, 41)
        x2 = torch.randn(1, 14, 21, 41)
        out1 = emulator.predict(x1)
        out2 = emulator.predict(x2)
        assert not torch.allclose(out1, out2)

    def test_zero_input_gives_finite_output(self, emulator):
        out = emulator.predict(torch.zeros(1, 14, 21, 41))
        assert torch.all(torch.isfinite(out))

    def test_output_range_physically_plausible(self, emulator, standard_input):
        """
        After applying the per-bin ENERGY_CENTER offset the reconstructed
        log10-flux should be negative and above -50 (both physically motivated
        bounds for diffuse galactic gamma-ray emission).
        """
        out = emulator.predict(standard_input).numpy()
        for i, offset in enumerate(ENERGY_CENTER):
            log_flux = out[0, i] + offset  # (H, W) map
            assert np.all(log_flux < 0), (
                f"Energy bin {i}: positive log-flux is unphysical"
            )
            assert np.all(log_flux > -50), (
                f"Energy bin {i}: log-flux below -50 is suspiciously extreme"
            )

    def test_physical_flux_bins_ordered_by_decreasing_value(self, emulator, standard_input):
        """
        After applying ENERGY_CENTER the reconstructed physical log-flux should
        decrease across energy bins on average. The ordering is guaranteed by
        ENERGY_CENTER (tested in test_schema.py), but this verifies that the
        model output does not catastrophically invert that ordering.
        """
        out = emulator.predict(standard_input)[0].numpy()  # (7, H, W)
        energy_center = ENERGY_CENTER  # shape (7,)
        # Mean physical log-flux per bin: model_output + offset
        mean_physical = [
            float(out[i].mean()) + energy_center[i] for i in range(7)
        ]
        diffs = [mean_physical[i + 1] - mean_physical[i] for i in range(6)]
        # The physical spectrum must be predominantly decreasing
        n_decreasing = sum(d < 0 for d in diffs)
        assert n_decreasing >= 4, (
            f"Expected ≥4 of 6 consecutive bin pairs to be decreasing in "
            f"physical log-flux space, got {n_decreasing}. Values: {mean_physical}"
        )


# ---------------------------------------------------------------------------
# Error handling and edge cases
# ---------------------------------------------------------------------------


class TestPredictEdgeCases:

    def test_wrong_channel_count_raises(self, emulator):
        """First Conv2d expects 14 input channels."""
        x = torch.randn(1, 7, 21, 41)
        with pytest.raises(RuntimeError):
            emulator.predict(x)

    def test_float64_input_raises(self, emulator):
        """Model is float32; float64 input causes a dtype mismatch in Conv2d."""
        x = torch.randn(1, 14, 21, 41, dtype=torch.float64)
        with pytest.raises(RuntimeError):
            emulator.predict(x)

    def test_numpy_array_raises(self, emulator):
        """predict() calls .to(device) on the tensor — numpy arrays have no .to()."""
        x = np.zeros((1, 14, 21, 41), dtype=np.float32)
        with pytest.raises((AttributeError, TypeError)):
            emulator.predict(x)

    def test_missing_batch_dim_raises(self, emulator):
        """Input must be 4-D (B, C, H, W). A 3-D tensor should be rejected."""
        x = torch.randn(14, 21, 41)
        with pytest.raises((RuntimeError, ValueError)):
            emulator.predict(x)

    def test_spatial_dims_too_small_raises(self, emulator):
        """
        Spatial dims of 8 collapse to 0 after 4 MaxPool2d(2,2) layers:
            8 → 4 → 2 → 1 → 0  (RuntimeError)
        """
        x = torch.randn(1, 14, 8, 8)
        with pytest.raises(RuntimeError):
            emulator.predict(x)

    def test_nan_input_produces_non_finite_output(self, emulator):
        """NaN propagates through the network — document this behaviour explicitly."""
        x = torch.randn(1, 14, 21, 41)
        x[0, 0, 0, 0] = float("nan")
        out = emulator.predict(x)
        assert torch.any(~torch.isfinite(out)), (
            "A NaN in the input should propagate to at least some output pixels"
        )

    def test_inf_input_produces_non_finite_output(self, emulator):
        """Inf likewise propagates through the network."""
        x = torch.randn(1, 14, 21, 41)
        x[0, 0, 0, 0] = float("inf")
        out = emulator.predict(x)
        assert torch.any(~torch.isfinite(out))

    def test_empty_batch_raises_or_produces_empty_output(self, emulator):
        """
        A batch of size 0 is degenerate. PyTorch either raises or returns an
        empty tensor — neither should silently produce wrong-shaped output.
        """
        x = torch.randn(0, 14, 21, 41)
        try:
            out = emulator.predict(x)
            # If it doesn't raise, the output must also be empty or have shape (0,...)
            assert out.shape[0] == 0
        except (RuntimeError, ValueError):
            pass  # Raising is also acceptable
