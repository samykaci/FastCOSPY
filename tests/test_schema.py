"""
Tests for fastcospy/schema.py.

The schema defines the feature engineering contract between the user's
preprocessing pipeline and the neural network. Any silent change to channel
order, normalization constants, or output energy offsets would silently break
physical predictions, so every constant is pinned here.
"""

import numpy as np
import pytest

from fastcospy.schema import (
    B_MAX,
    D_MAX,
    ENERGY_CENTER,
    EPSILON,
    GAS_INTERCEPT,
    GAS_SCALE,
    GLOBAL_SOURCE_INTERCEPT,
    INPUT_CHANNELS,
    LOCAL_SOURCE_INTERCEPT,
)


# ---------------------------------------------------------------------------
# Input channels
# ---------------------------------------------------------------------------


class TestInputChannels:
    """The model was trained with exactly 14 ordered input channels."""

    def test_count_is_exactly_14(self):
        assert len(INPUT_CHANNELS) == 14

    def test_all_entries_are_strings(self):
        for i, ch in enumerate(INPUT_CHANNELS):
            assert isinstance(ch, str), f"Channel at index {i} is not a string: {ch!r}"

    def test_no_empty_names(self):
        for i, ch in enumerate(INPUT_CHANNELS):
            assert ch.strip(), f"Empty or whitespace-only channel name at index {i}"

    def test_names_are_unique(self):
        assert len(set(INPUT_CHANNELS)) == len(INPUT_CHANNELS), (
            "Duplicate channel names detected"
        )

    def test_is_list_or_tuple(self):
        assert isinstance(INPUT_CHANNELS, (list, tuple))

    # Exact channel positions — these are fixed by training and must not shift.
    @pytest.mark.parametrize(
        "index, name",
        [
            (0, "cos_l"),
            (1, "sin_l"),
            (2, "b_over_BMAX"),
            (3, "log10_halo_size"),
            (13, "log10_gas_density"),
        ],
    )
    def test_channel_at_expected_index(self, index, name):
        assert INPUT_CHANNELS[index] == name, (
            f"Channel {index}: expected {name!r}, got {INPUT_CHANNELS[index]!r}"
        )

    def test_geometric_channels_come_first(self):
        """The first four channels encode sky position and halo geometry."""
        assert INPUT_CHANNELS[0] == "cos_l"
        assert INPUT_CHANNELS[1] == "sin_l"
        assert INPUT_CHANNELS[2] == "b_over_BMAX"
        assert INPUT_CHANNELS[3] == "log10_halo_size"

    def test_gas_density_is_last(self):
        assert INPUT_CHANNELS[-1] == "log10_gas_density"

    def test_source_channels_in_middle(self):
        """Source-related features occupy indices 4–12."""
        source_block = INPUT_CHANNELS[4:13]
        assert len(source_block) == 9
        # Spot-check a few expected names
        assert "mean_gas" in source_block
        assert "std_gas" in source_block


# ---------------------------------------------------------------------------
# Energy center offsets
# ---------------------------------------------------------------------------


class TestEnergyCenter:
    """
    ENERGY_CENTER encodes per-bin flux offsets used to convert model output
    back to physical log10-flux space:
        log_flux_physical = model_output[i] + ENERGY_CENTER[i]
    """

    def test_count_is_exactly_7(self):
        assert len(ENERGY_CENTER) == 7

    def test_is_numpy_array(self):
        assert isinstance(ENERGY_CENTER, np.ndarray)

    def test_dtype_is_floating(self):
        assert np.issubdtype(ENERGY_CENTER.dtype, np.floating)

    def test_no_nans_or_infs(self):
        assert np.all(np.isfinite(ENERGY_CENTER)), "ENERGY_CENTER must not contain NaN or Inf"

    def test_all_values_are_negative(self):
        """Physical log10(flux) values for diffuse gamma rays are always negative."""
        assert np.all(ENERGY_CENTER < 0), (
            "All ENERGY_CENTER values should be negative (they represent tiny log-fluxes)"
        )

    def test_monotonically_decreasing(self):
        """
        Higher-energy bins correspond to systematically lower fluxes.
        This must be strictly decreasing to avoid silent index mixups.
        """
        diffs = np.diff(ENERGY_CENTER)
        assert np.all(diffs < 0), (
            f"ENERGY_CENTER must be strictly decreasing, got diffs={diffs}"
        )

    def test_physically_plausible_range(self):
        """
        Diffuse galactic gamma-ray fluxes sit in roughly 10^-30 to 10^-15
        eV^-1 m^-2 s^-1 sr^-1, so offsets outside [-35, -10] are suspicious.
        """
        assert np.all(ENERGY_CENTER >= -35), "Offsets below -35 seem unphysical"
        assert np.all(ENERGY_CENTER <= -10), "Offsets above -10 seem unphysical"

    # Pin every bin to guard against accidental reordering or value drift.
    @pytest.mark.parametrize(
        "index, value",
        [
            (0, -20.5),
            (1, -21.5),
            (2, -22.0),
            (3, -23.0),
            (4, -23.5),
            (5, -24.5),
            (6, -26.0),
        ],
    )
    def test_bin_offset_exact_value(self, index, value):
        assert ENERGY_CENTER[index] == pytest.approx(value), (
            f"ENERGY_CENTER[{index}] changed from {value}"
        )

    def test_energy_span_is_5_5_decades(self):
        """Energy bins span from -20.5 to -26.0, a total spread of 5.5."""
        span = ENERGY_CENTER[0] - ENERGY_CENTER[-1]
        assert span == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------


class TestNormalizationConstants:
    """
    These constants are baked into the training preprocessing. Changing any
    of them would silently invalidate all predictions made with v1.0.0 weights.
    """

    def test_b_max_value(self):
        assert B_MAX == pytest.approx(5.0)

    def test_b_max_positive(self):
        assert B_MAX > 0

    def test_d_max_value(self):
        assert D_MAX == pytest.approx(32.0)

    def test_d_max_positive(self):
        assert D_MAX > 0

    def test_epsilon_value(self):
        assert EPSILON == pytest.approx(1.0)

    def test_epsilon_positive(self):
        """EPSILON is added before log10 to avoid log(0). Must be > 0."""
        assert EPSILON > 0

    def test_gas_scale_value(self):
        assert GAS_SCALE == pytest.approx(3.5)

    def test_gas_scale_positive(self):
        assert GAS_SCALE > 0

    def test_gas_intercept_value(self):
        assert GAS_INTERCEPT == pytest.approx(-4.5)

    def test_local_source_intercept_value(self):
        assert LOCAL_SOURCE_INTERCEPT == pytest.approx(1.5)

    def test_global_source_intercept_value(self):
        assert GLOBAL_SOURCE_INTERCEPT == pytest.approx(1.3)

    def test_all_constants_are_finite(self):
        for name, val in [
            ("B_MAX", B_MAX),
            ("D_MAX", D_MAX),
            ("EPSILON", EPSILON),
            ("GAS_SCALE", GAS_SCALE),
            ("GAS_INTERCEPT", GAS_INTERCEPT),
            ("LOCAL_SOURCE_INTERCEPT", LOCAL_SOURCE_INTERCEPT),
            ("GLOBAL_SOURCE_INTERCEPT", GLOBAL_SOURCE_INTERCEPT),
        ]:
            assert np.isfinite(val), f"{name} is not finite"
