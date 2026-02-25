# FastCOSPY AI Coding Guidelines

## Architecture Overview
FastCOSPY implements a U-Net-based neural network with LSTM and spatial attention for multi-energy image prediction. The model processes 2D images through an encoder-decoder architecture, applies attention at the bottleneck, and uses LSTM for sequence prediction across 7 energy levels.

Key components in `model.py`:
- **DoubleConvolution**: Standard two-layer conv block with GroupNorm (groups=8) and optional dropout
- **Down/Up**: Encoder/decoder blocks with max pooling and bilinear upsampling
- **SpatialAttentionGate**: Attention mechanism using gating signals
- **UNetLSTMAttention**: Main model class orchestrating the full pipeline

## Core Patterns
- **Normalization**: Always use `nn.GroupNorm(GN_GROUPS, out_channels)` after convolutions, where `GN_GROUPS = 8`
- **Upsampling**: Use bilinear interpolation with `align_corners=True` for skip connection concatenation
- **Skip Connections**: Pad upsampled features to match skip tensor dimensions before concatenation
- **Tensor Reshaping**: Flatten spatial dims, repeat for energy levels, then reshape back (see forward method)
- **LSTM Integration**: Process flattened features as sequences, predict per-energy outputs

## Model Configuration
- Default channels: [96, 192, 384, 768, 1536] for encoder stages
- Latent channels: 128 (decoder output)
- LSTM hidden size: 128
- Energy levels: 7 (configurable via `n_energies`)

## Development Workflow
- Environment: Use conda for Python environment management (configured in `.vscode/settings.json`)
- Dependencies: PyTorch with CUDA support assumed
- Testing: No automated tests visible; validate model outputs manually
- Debugging: Check tensor shapes at each stage, especially after reshaping operations

## Code Style
- Follow PyTorch conventions for module definitions
- Use `nn.Sequential` for multi-layer blocks
- Document parameters and return types in docstrings
- Use `inplace=True` for ReLU operations where safe

## Common Modifications
- To change energy levels: Update `n_energies` parameter and adjust downstream processing
- Adding attention: Follow `SpatialAttentionGate` pattern with gating and feature branches
- Modifying architecture: Maintain channel progression ratios (doubling/halving)