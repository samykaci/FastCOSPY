"""
FastCOSPY schema.

Defines:
- The 14 input channels in order
- The fixed output rescaling parameters (scale & shift)
"""

import numpy as np

# ---------------------------
# Input channels (C = 14)
# ---------------------------
INPUT_CHANNELS = [
    "cos_l",                       # cos(Galactic longitude)
    "sin_l",                       # sin(Galactic longitude)
    "b_over_BMAX",                 # Normalized Galactic latitude
    "log10_halo_size",             # Log of diffusion halo size
    "log10_total_source_count_plus_LOCAL_SOURCE_INTERCEPT",    # Log-sum of source contributions
    "mean_gas",                    # Mean gas density (weighted)
    "std_gas",                     # Standard deviation of gas density
    "log10_max_source_count_plus_LOCAL_SOURCE_INTERCEPT",      # Max source contribution
    "gas_max",                     # Gas density of dominant source
    "log10_second_max_source_plus_LOCAL_SOURCE_INTERCEPT",     # Second-largest source contribution
    "gas_max_2",                   # Gas density of second-dominant source
    "min_source_distance_over_D_MAX",         # Min source distance (normalized)
    "log10_mean_total_sources_plus_GLOBAL_SOURCE_INTERCEPT",    # Mean source contribution over the entire Galaxy
    "log10_gas_density",           # Line-of-sight gas density
]
# The gas of sources comes as source_gas = (np.log10(gas + EPSILON) + GAS_INTERCEPT) / GAS_SCALE
#The gas of the line of sight is just the log normalized between 0 and 1.
#----------------------------
# Scaling parameters for the input channels (for normalization)
#----------------------------

B_MAX = 5.
LOCAL_SOURCE_INTERCEPT = 1.5
GLOBAL_SOURCE_INTERCEPT = 1.3
D_MAX = 32.
EPSILON = 1.
GAS_INTERCEPT = -4.5
GAS_SCALE = 3.5

# ---------------------------
# Output energies (E = 7)
# ---------------------------

# ---------------------------
# Output rescaling
# ---------------------------
# log_flux_physical = output_scaled + ENERGY_CENTER
ENERGY_CENTER = np.array([-20.5, -21.5, -22., -23, -23.5, -24.5, -26.])
# Note that there is no slope for the rescaling.