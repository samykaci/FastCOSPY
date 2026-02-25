# FastCOSPY

FastCOSPY is an open-source, AI-powered surrogate model for the COSPY (private) framework,
designed to deliver gamma-ray flux predictions in approximately **0.25 seconds
on CPU**.

It is intended to be integrated into existing simulation or analysis pipelines
as a **drop-in inference accelerator**, replacing expensive numerical solvers
once the inputs have been properly prepared.

FastCOSPY **does not perform physical preprocessing**. It assumes that inputs
are already transformed into the feature representation expected by the model based on the user's purpose, choice of diffusion mechanism, gas map, etc.

---

## Scope and philosophy

FastCOSPY focuses on **fast inference only**.

- Neural-network inference
- Designed for easy integration into existing codes

This design choice allows FastCOSPY to remain **agnostic to upstream modeling
choices** and usable in a wide variety of workflows.

---

## Features

- CPU-compatible neural-network emulator (GPU optional)
- Orders-of-magnitude faster than full physical simulations
- Minimal, inference-only API
- Designed for integration into existing physics or ML pipelines
- Deterministic, reproducible predictions

---

## Installation

### From PyPI

```bash
pip install fastcospy
```

### Requirements

- Python ≥ 3.9
- NumPy
- PyTorch (CPU or GPU)

---

## Core usage

```python
import numpy as np
from fastcospy import FastCOSPYEmulator

# Initialize the emulator
model = FastCOSPYEmulator(device="cpu")

# Preprocessed feature tensor
# Input: `torch.Tensor` of shape (B, C, latitude, longitude)
# Note in the provided example, input features are prepared as a NumPy array and converted to a torch.Tensor before inference.

features = ...

# Predict gamma-ray flux maps
flux_maps = model.predict(features)
```

FastCOSPY assumes that `features`:
- Are already normalized and formatted correctly
- Match the feature definition used during training
- Lie within the training domain of the model

---

## Input format

The emulator expects a torch.Tensor array with shape:

```
(B, C, latitude, longitude)
```

where `B` is the number of pictures per batch and `C` is the number of feature channels.

The **meaning, scaling, and construction of these features are user-defined**
and external to FastCOSPY.

---

## Architecture

The FastCOSPY emulator is based on a hybrid architecture combining a U-Net, an LSTM and an attention gate. The U-Net handles the prediction of the spatial structure of the output flux maps. The LSTM models the energy sequence and the attention gate enhances relevant spatial features at the bottleneck of the U-Net. Unlike transformers, this architecture leads to manageable training without overfitting the available data. At the same time, it produces a sufficiently accurate model that can run on most machines.

---

### Input Feature Channels

FastCOSPY expects a **14-channel feature tensor** of shape:

    (B, C=14, latitude, longitude)

Each channel has a fixed semantic meaning and normalization (see schema.py).

| Channel | Name                      | Description |
|--------:|---------------------------|------------|
| 0 | cos(l) | Cosine of Galactic longitude |
| 1 | sin(l) | Sine of Galactic longitude |
| 2 | b / B_max | Normalized Galactic latitude |
| 3 | log10(halo_size) | Logarithm of diffusion halo size |
| 4 | log10(total_source_count) | Log-sum of source contributions |
| 5 | mean_gas | Mean gas density weighted by source contribution |
| 6 | std_gas | Standard deviation of gas density |
| 7 | log10(max_source_count) | Maximum source contribution |
| 8 | gas_max | Gas density of dominant source |
| 9 | log10(second_max_source_count) | Second-largest source contribution |
| 10 | gas_max_2 | Gas density of second-dominant source |
| 11 | min_source_distance / D_MAX | Minimum source distance (normalized) |
| 12 | log10(mean_total_sources) | Mean integrated source contribution |
| 13 | gas_density | Logarithm of line-of-sight gas density |

---

## Output format

The output is a torch.Tensor array with shape:

```
(B, energy_bins=7, latitude, longitude)
```

Each slice in energy corresponds to a predicted gamma-ray flux map at that energy. Outputs are provided in logarithmic space at 7 energies. The 7 energy bins are logarithmically spaced between 1e13 eV and 1e15 eV. Further energy-dependent shifting of the output is needed using the parameters provided (see schema.py).

---

## Example

A fully worked example is provided in the `example/` directory with a few lists of sources:

```
example/
├── gas_data.npy
├── lists_of_sources.npz
└── example_script.py
```

The example demonstrates:
- How a `lists_of_sources` dataset can be converted into model-ready features
- How to call FastCOSPY for inference
- How to visualize the predicted flux maps

**Important**  
The feature construction shown in the example is **illustrative only**.
It is not part of the FastCOSPY API and must be adapted or replaced depending
on the user's pipeline.

---

## Performance

Typical inference time:

- 95% of predictions lie within 16% relative error in logarithmic flux space. Tests have been performed on a geometry and diffusion mechanism never used during training, signaling strong generalization.
- **~0.25 seconds per full-sky prediction on CPU**
- Faster on GPU (optional)

Performance depends on hardware.

---

## Limitations

- Valid only within the training domain
- Performance may degrade for extreme diffusion configurations
- Not a physical simulator
- Requires correctly normalized inputs

---

## Docker

A CPU-only Docker image is provided for reproducibility.

### Build the image

```bash
docker build -t fastcospy .
```

### Run the container

```bash
docker run --rm fastcospy
```

GPU support is optional and should be configured by the user if needed.

---

## Intended use

FastCOSPY is designed for:
- Rapid parameter scans
- AI acceleration
- Surrogate-based inference
- ML-augmented physics workflows

It is **not** intended as a replacement for physical simulations outside the
domain covered by its training data.

---

## Disclaimer

FastCOSPY is a surrogate model trained on COSPY simulation outputs.
Predictions are valid only within the training regime and should not be
extrapolated without validation.

This software is provided for research purposes only.

---

## License

MIT License
