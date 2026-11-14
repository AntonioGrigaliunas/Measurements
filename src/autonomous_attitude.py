"""Autonomous self-reconfiguring CubeSat attitude-determination logic.

Operating modes
---------------
NORMAL
    Sun array and magnetometer are accepted; full TRIAD attitude is available.
RECONFIGURED
    One impactful Sun-sensor failure is detected and the Sun vector is
    reconstructed using the remaining channels; full TRIAD attitude remains
    available.
SUN_SAFE
    A usable Sun direction is available, but the magnetometer fails the
    magnetic-health checks. Full 3D attitude is intentionally withheld rather
    than reporting a potentially corrupted TRIAD solution.
SAFE
    A trustworthy Sun direction cannot be produced.

The hard-fault logic and the fuzzy reliability layer have different jobs:
hard-fault logic protects/reconfigures the estimator, while fuzzy logic can
remain an interpretable confidence output.
"""

from __future__ import annotations

import numpy as np

from attitude_estimation import (
    angular_error_degrees,
    estimate_sun_vector,
    estimate_sun_vector_reconfigured,
    triad_attitude,
)
from sensor_parameters import SUN_GAINS, SUN_OFFSETS
from sun_sensor import unit_vector
from sun_sensor_fault_detection import detect_failed_sun_sensor


MAG_NORM_ERROR_LIMIT = 0.0155
MAG_CONSISTENCY_ERROR_LIMIT_DEG = 5.75


def assess_magnetometer_health(
    estimated_sun_body: np.ndarray,
    magnetic_measurement: np.ndarray,
    sun_reference: np.ndarray,
    magnetic_reference: np.ndarray,
    norm_error_limit: float = MAG_NORM_ERROR_LIMIT,
    consistency_error_limit_deg: float = MAG_CONSISTENCY_ERROR_LIMIT_DEG,
) -> dict[str, float | bool]:
    """Assess whether the magnetometer is suitable for full TRIAD attitude."""
    magnetic_direction = unit_vector(magnetic_measurement)

    reference_separation = angular_error_degrees(
        sun_reference,
        magnetic_reference,
    )
    measured_separation = angular_error_degrees(
        estimated_sun_body,
        magnetic_direction,
    )

    consistency_error = abs(
        measured_separation - reference_separation
    )
    norm_error = abs(np.linalg.norm(magnetic_measurement) - 1.0)

    healthy = (
        norm_error <= norm_error_limit
        and consistency_error <= consistency_error_limit_deg
    )

    return {
        "healthy": bool(healthy),
        "norm_error": float(norm_error),
        "consistency_error_deg": float(consistency_error),
    }


def autonomous_attitude_solution(
    sensor_normals: np.ndarray,
    sun_readings: np.ndarray,
    magnetic_measurement: np.ndarray,
    sun_reference: np.ndarray,
    magnetic_reference: np.ndarray,
    gains: np.ndarray = SUN_GAINS,
    offsets: np.ndarray = SUN_OFFSETS,
) -> dict[str, object]:
    """Return the autonomous attitude solution and current operating mode."""
    detection = detect_failed_sun_sensor(
        sun_readings,
        magnetic_measurement,
        sun_reference,
        magnetic_reference,
        gains,
        offsets,
    )
    failed_sensor = detection["failed_sensor"]

    try:
        if failed_sensor is None:
            estimated_sun_body = estimate_sun_vector(
                sensor_normals,
                sun_readings,
            )
            sun_reconfigured = False
        else:
            healthy_mask = np.ones(len(sun_readings), dtype=bool)
            healthy_mask[int(failed_sensor)] = False
            estimated_sun_body = estimate_sun_vector_reconfigured(
                sensor_normals,
                sun_readings,
                healthy_mask,
                gains=gains,
                offsets=offsets,
            )
            sun_reconfigured = True
    except ValueError:
        return {
            "mode": "SAFE",
            "rotation": None,
            "sun_body": None,
            "magnetic_body": None,
            "detected_failed_sun_sensor": failed_sensor,
            "sun_fault_detection": detection,
            "magnetometer_health": None,
        }

    magnetic_health = assess_magnetometer_health(
        estimated_sun_body,
        magnetic_measurement,
        sun_reference,
        magnetic_reference,
    )

    if not bool(magnetic_health["healthy"]):
        return {
            "mode": "SUN_SAFE",
            "rotation": None,
            "sun_body": estimated_sun_body,
            "magnetic_body": unit_vector(magnetic_measurement),
            "detected_failed_sun_sensor": failed_sensor,
            "sun_fault_detection": detection,
            "magnetometer_health": magnetic_health,
        }

    estimated_magnetic_body = unit_vector(magnetic_measurement)
    try:
        rotation = triad_attitude(
            sun_reference,
            magnetic_reference,
            estimated_sun_body,
            estimated_magnetic_body,
        )
    except ValueError:
        return {
            "mode": "SUN_SAFE",
            "rotation": None,
            "sun_body": estimated_sun_body,
            "magnetic_body": estimated_magnetic_body,
            "detected_failed_sun_sensor": failed_sensor,
            "sun_fault_detection": detection,
            "magnetometer_health": magnetic_health,
        }

    return {
        "mode": "RECONFIGURED" if sun_reconfigured else "NORMAL",
        "rotation": rotation,
        "sun_body": estimated_sun_body,
        "magnetic_body": estimated_magnetic_body,
        "detected_failed_sun_sensor": failed_sensor,
        "sun_fault_detection": detection,
        "magnetometer_health": magnetic_health,
    }
