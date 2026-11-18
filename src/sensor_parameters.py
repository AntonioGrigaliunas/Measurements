"""Shared sensor parameters and simulation assumptions."""

from __future__ import annotations

import numpy as np


# Field magnitude used for normalization.
EARTH_FIELD_UT = 50.0
EARTH_FIELD_NT = EARTH_FIELD_UT * 1000.0

# Sun-sensor parameters.
SUN_NOISE_STD = 0.02
SUN_GAINS = np.array([1.03, 0.98, 0.96, 1.02, 1.05, 0.97])
SUN_OFFSETS = np.array([0.010, 0.005, 0.000, 0.008, 0.012, 0.004])

# Magnetometer parameters, normalized by 50 microtesla.
RM3100_NOISE_NT = 17.0
RM3100_BIAS_NT = 440.0
RM3100_SCALE_FACTOR_ERROR = 4885e-6

MAG_NOISE_STD = RM3100_NOISE_NT / EARTH_FIELD_NT
MAG_BIAS = np.array(
    [
        RM3100_BIAS_NT / EARTH_FIELD_NT,
        -0.70 * RM3100_BIAS_NT / EARTH_FIELD_NT,
        0.45 * RM3100_BIAS_NT / EARTH_FIELD_NT,
    ]
)
MAG_SCALE_FACTORS = np.array(
    [
        1.0 + RM3100_SCALE_FACTOR_ERROR,
        1.0 - RM3100_SCALE_FACTOR_ERROR,
        1.0 + 0.60 * RM3100_SCALE_FACTOR_ERROR,
    ]
)

# Fixed magnetic-disturbance direction.
MAG_DISTURBANCE_DIRECTION = np.array([0.15, -0.10, 0.08], dtype=float)
MAG_DISTURBANCE_DIRECTION = (
    MAG_DISTURBANCE_DIRECTION
    / np.linalg.norm(MAG_DISTURBANCE_DIRECTION)
)

# Representative 7.5 microtesla magnetic disturbance.
MAG_FAULT_UT = 7.5
MAG_FAULT_NORMALIZED = MAG_FAULT_UT / EARTH_FIELD_UT
MAG_FAULT_VECTOR = MAG_FAULT_NORMALIZED * MAG_DISTURBANCE_DIRECTION
