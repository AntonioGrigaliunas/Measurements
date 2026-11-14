"""Automatic Sun-sensor fault detection for the self-reconfiguring CubeSat.

The detector exploits the six-face +/-X, +/-Y, +/-Z geometry.

A hard failure that matters most occurs when the illuminated sensor of an
opposing pair reports approximately zero. In that case both sensors in the pair
appear dark even though the other two axes imply, through norm(s) = 1, that the
missing component should be substantial.

The sign ambiguity of that missing component is resolved using the known angle
between the Sun and geomagnetic reference vectors and the measured
magnetometer direction.

Important limitation
--------------------
A failed sensor that is already on the dark side of the CubeSat is not
observable from one static snapshot because zero is also its correct reading.
Such a dormant failure has little immediate effect on the Sun estimate and can
be detected later when spacecraft motion brings that face into illumination.
"""

from __future__ import annotations

import numpy as np

from attitude_estimation import angular_error_degrees
from sun_sensor import unit_vector


DEFAULT_DARK_THRESHOLD = 0.05
DEFAULT_MIN_MISSING_COMPONENT = 0.30
DEFAULT_MIN_FAILED_RESIDUAL = 0.15
DEFAULT_MAX_CONSISTENCY_ERROR_DEG = 6.0


def detect_failed_sun_sensor(
    readings: np.ndarray,
    magnetic_measurement: np.ndarray,
    sun_reference: np.ndarray,
    magnetic_reference: np.ndarray,
    gains: np.ndarray,
    offsets: np.ndarray,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    min_missing_component: float = DEFAULT_MIN_MISSING_COMPONENT,
    min_failed_residual: float = DEFAULT_MIN_FAILED_RESIDUAL,
    max_consistency_error_deg: float = DEFAULT_MAX_CONSISTENCY_ERROR_DEG,
) -> dict[str, float | int | str | None]:
    """Detect one impactful hard-failed Sun-sensor channel."""
    readings = np.asarray(readings, dtype=float)
    magnetic_measurement = np.asarray(magnetic_measurement, dtype=float)
    gains = np.asarray(gains, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    if readings.shape != (6,):
        raise ValueError("readings must contain 6 Sun-sensor values.")
    if gains.shape != (6,) or offsets.shape != (6,):
        raise ValueError("gains and offsets must each contain 6 values.")

    corrected = np.maximum(0.0, (readings - offsets) / gains)
    signed_components = np.array(
        [
            corrected[0] - corrected[1],
            corrected[2] - corrected[3],
            corrected[4] - corrected[5],
        ]
    )

    reference_separation = angular_error_degrees(
        sun_reference,
        magnetic_reference,
    )
    magnetic_direction = unit_vector(magnetic_measurement)

    pairs = [(0, 1), (2, 3), (4, 5)]
    candidates: list[dict[str, float | int]] = []

    for axis, (positive_index, negative_index) in enumerate(pairs):
        if max(corrected[positive_index], corrected[negative_index]) > dark_threshold:
            continue

        known_axes = [index for index in range(3) if index != axis]
        known_squared = float(np.sum(signed_components[known_axes] ** 2))
        missing_magnitude = np.sqrt(max(0.0, 1.0 - known_squared))

        if missing_magnitude < min_missing_component:
            continue

        for sign, sensor_index in (
            (1.0, positive_index),
            (-1.0, negative_index),
        ):
            sun_candidate = signed_components.copy()
            sun_candidate[axis] = sign * missing_magnitude
            sun_candidate = unit_vector(sun_candidate)

            measured_separation = angular_error_degrees(
                sun_candidate,
                magnetic_direction,
            )
            consistency_error = abs(
                measured_separation - reference_separation
            )

            failed_residual = missing_magnitude - corrected[sensor_index]
            if failed_residual < min_failed_residual:
                continue

            candidates.append(
                {
                    "sensor_index": sensor_index,
                    "axis": axis,
                    "missing_component": float(missing_magnitude),
                    "failed_residual": float(failed_residual),
                    "consistency_error_deg": float(consistency_error),
                }
            )

    if not candidates:
        return {
            "failed_sensor": None,
            "status": "no_impactful_failure_detected",
            "consistency_error_deg": None,
            "missing_component": None,
            "failed_residual": None,
        }

    best = min(
        candidates,
        key=lambda candidate: float(candidate["consistency_error_deg"]),
    )

    if float(best["consistency_error_deg"]) > max_consistency_error_deg:
        return {
            "failed_sensor": None,
            "status": "candidate_rejected_by_vector_consistency",
            "consistency_error_deg": float(best["consistency_error_deg"]),
            "missing_component": float(best["missing_component"]),
            "failed_residual": float(best["failed_residual"]),
        }

    return {
        "failed_sensor": int(best["sensor_index"]),
        "status": "impactful_failure_detected",
        "consistency_error_deg": float(best["consistency_error_deg"]),
        "missing_component": float(best["missing_component"]),
        "failed_residual": float(best["failed_residual"]),
    }
