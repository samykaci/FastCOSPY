"""
FastCOSPY package

Provides the FastCOSPY U-Net+LSTM+Attention model and an emulator wrapper for direct usage.

Modules
-------
models : Contains the PyTorch model definitions (UNetLSTMAttention, DoubleConvolution...)
emulator : High-level interface for running the model without dealing with PyTorch tensors.
"""

from .models import UNetLSTMAttention, DoubleConvolution, Down, Up, SpatialAttentionGate
from .emulator import FastCOSPYEmulator

__all__ = ["UNetLSTMAttention", "DoubleConvolution", "Down", "Up", "SpatialAttentionGate", "FastCOSPYEmulator"]

__version__ = "1.0.0"
__author__ = "Samy Kaci"