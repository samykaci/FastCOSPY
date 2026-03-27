"""
Shared fixtures and pytest configuration for the FastCOSPY test suite.

Custom markers
--------------
network : test makes real HTTP requests (weight download from GitHub Releases).
          Skip with: pytest -m "not network"
slow    : test is computationally expensive (e.g. full forward pass on large inputs).
          Skip with: pytest -m "not slow"

Session-scoped emulator
-----------------------
The `emulator` fixture downloads pretrained weights exactly once per test session
and reuses the cache (~/.fastcospy/) on subsequent runs. Tests that depend on this
fixture will make a network request on a cold cache but are otherwise offline.
"""

import pytest
import torch

from fastcospy import FastCOSPYEmulator, UNetLSTMAttention


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "network: test requires a network connection (weight download from GitHub Releases)",
    )
    config.addinivalue_line(
        "markers",
        "slow: test is computationally expensive",
    )


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def emulator() -> FastCOSPYEmulator:
    """
    Session-scoped FastCOSPYEmulator with pretrained weights.

    Weights are downloaded once and cached in ~/.fastcospy. Subsequent test
    runs reuse the cache without any network access.
    """
    return FastCOSPYEmulator(device="cpu")


@pytest.fixture(scope="module")
def raw_model() -> UNetLSTMAttention:
    """
    Module-scoped UNetLSTMAttention with random weights (no download).

    Suitable for testing shapes, gradient flow, and architecture properties
    without any network dependency.
    """
    model = UNetLSTMAttention(n_channels=14)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tensor fixtures  (function-scoped for isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
def standard_input() -> torch.Tensor:
    """Single-sample input matching training patch dimensions (B=1, C=14, H=21, W=41)."""
    torch.manual_seed(0)
    return torch.randn(1, 14, 21, 41)


@pytest.fixture
def batch_input() -> torch.Tensor:
    """Batch of 4 samples (B=4, C=14, H=21, W=41)."""
    torch.manual_seed(1)
    return torch.randn(4, 14, 21, 41)
