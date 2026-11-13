"""Central sensor assumptions for the MA4822 CubeSat simulation.

Hardware basis
--------------
Sun sensor:
- Six-face cosine-response coarse Sun-sensor architecture.
- Representative coarse cosine Sun-sensor accuracy: about 5 deg (3-sigma).
- A 5 deg (3-sigma) angular uncertainty maps to roughly 0.02 normalized
  1-sigma output noise near 45 deg incidence, so SUN_NOISE_STD = 0.02.

Magnetometer:
- PNI RM3100-class 3-axis magnetometer.
- Representative nanosatellite values:
    noise: 17 nT
    worst-axis bias: 440 nT
    worst-axis scale-factor error: 4885 ppm
- Earth's field is normalized using a representative 50 microtesla magnitude.

The Sun-sensor gain/offset vectors below are residual calibration assumptions
for sensitivity analysis; they are not claimed as vendor datasheet values.
"""

from __future__ import annotations

import numpy as np


# Representative geomagnetic field used to normalize magnetometer quantities.
EARTH_FIELD_UT = 50.0
EARTH_FIELD_NT = EARTH_FIELD_UT * 1000.0

# Coarse Sun-sensor model.
SUN_NOISE_STD = 0.02
SUN_GAINS = np.array([1.03, 0.98, 0.96, 1.02, 1.05, 0.97])
SUN_OFFSETS = np.array([0.010, 0.005, 0.000, 0.008, 0.012, 0.004])

# RM3100-class magnetometer parameters, normalized by 50 microtesla.
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

# Fixed disturbance direction used for magnetic-fault studies.
MAG_DISTURBANCE_DIRECTION = np.array([0.15, -0.10, 0.08], dtype=float)
MAG_DISTURBANCE_DIRECTION = (
    MAG_DISTURBANCE_DIRECTION
    / np.linalg.norm(MAG_DISTURBANCE_DIRECTION)
)

# Representative fault case: 7.5 microtesla disturbance, comparable in scale
# to published CubeSat onboard magnetic interference measurements.
MAG_FAULT_UT = 7.5
MAG_FAULT_NORMALIZED = MAG_FAULT_UT / EARTH_FIELD_UT
MAG_FAULT_VECTOR = MAG_FAULT_NORMALIZED * MAG_DISTURBANCE_DIRECTION
