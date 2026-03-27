"""
Tests that verify the package namespace, public API surface, and metadata.

These tests guard against:
- Accidentally removing public symbols
- Breaking __all__ / actual-exports consistency
- Shipping a bad version string
"""

import importlib
import inspect

import pytest


# ---------------------------------------------------------------------------
# Top-level package importability
# ---------------------------------------------------------------------------


def test_package_is_importable():
    importlib.import_module("fastcospy")


def test_models_module_is_importable():
    importlib.import_module("fastcospy.models")


def test_emulator_module_is_importable():
    importlib.import_module("fastcospy.emulator")


def test_utils_module_is_importable():
    importlib.import_module("fastcospy.utils")


def test_schema_module_is_importable():
    importlib.import_module("fastcospy.schema")


# ---------------------------------------------------------------------------
# Version / authorship metadata
# ---------------------------------------------------------------------------


def test_version_attribute_exists():
    import fastcospy

    assert hasattr(fastcospy, "__version__")


def test_version_is_string():
    from fastcospy import __version__

    assert isinstance(__version__, str)


def test_version_has_semantic_format():
    from fastcospy import __version__

    parts = __version__.split(".")
    assert len(parts) == 3, f"Expected MAJOR.MINOR.PATCH, got {__version__!r}"
    assert all(p.isdigit() for p in parts), (
        f"Each version component must be numeric, got {__version__!r}"
    )


def test_version_is_non_empty():
    from fastcospy import __version__

    assert __version__.strip()


def test_author_attribute_exists():
    import fastcospy

    assert hasattr(fastcospy, "__author__")


def test_author_is_non_empty_string():
    from fastcospy import __author__

    assert isinstance(__author__, str)
    assert __author__.strip()


# ---------------------------------------------------------------------------
# __all__ consistency
# ---------------------------------------------------------------------------


def test_all_is_defined():
    import fastcospy

    assert hasattr(fastcospy, "__all__")


def test_all_is_list_or_tuple():
    import fastcospy

    assert isinstance(fastcospy.__all__, (list, tuple))


def test_all_is_non_empty():
    import fastcospy

    assert len(fastcospy.__all__) > 0


def test_all_exports_are_accessible_on_package():
    import fastcospy

    for name in fastcospy.__all__:
        assert hasattr(fastcospy, name), (
            f"{name!r} is listed in __all__ but is not accessible on the package"
        )


def test_all_exports_have_no_duplicates():
    import fastcospy

    names = list(fastcospy.__all__)
    assert len(names) == len(set(names)), "Duplicate entries found in __all__"


# ---------------------------------------------------------------------------
# Required public symbols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    [
        "UNetLSTMAttention",
        "DoubleConvolution",
        "Down",
        "Up",
        "SpatialAttentionGate",
        "FastCOSPYEmulator",
    ],
)
def test_public_symbol_importable_from_package(symbol):
    import fastcospy

    assert hasattr(fastcospy, symbol), f"{symbol!r} is not importable from fastcospy"


@pytest.mark.parametrize(
    "symbol",
    [
        "UNetLSTMAttention",
        "DoubleConvolution",
        "Down",
        "Up",
        "SpatialAttentionGate",
    ],
)
def test_model_symbols_importable_from_models_module(symbol):
    from fastcospy import models

    assert hasattr(models, symbol), f"{symbol!r} is not in fastcospy.models"


def test_emulator_importable_from_emulator_module():
    from fastcospy.emulator import FastCOSPYEmulator  # noqa: F401


# ---------------------------------------------------------------------------
# Type checks on public objects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    [
        "UNetLSTMAttention",
        "DoubleConvolution",
        "Down",
        "Up",
        "SpatialAttentionGate",
        "FastCOSPYEmulator",
    ],
)
def test_public_symbol_is_a_class(symbol):
    import fastcospy

    obj = getattr(fastcospy, symbol)
    assert inspect.isclass(obj), f"{symbol!r} should be a class"


def test_unet_is_nn_module_subclass():
    import torch.nn as nn
    from fastcospy import UNetLSTMAttention

    assert issubclass(UNetLSTMAttention, nn.Module)


def test_building_block_classes_are_nn_module_subclasses():
    import torch.nn as nn
    from fastcospy import DoubleConvolution, Down, SpatialAttentionGate, Up

    for cls in (DoubleConvolution, Down, Up, SpatialAttentionGate):
        assert issubclass(cls, nn.Module), f"{cls.__name__} must subclass nn.Module"
