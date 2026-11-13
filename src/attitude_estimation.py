"""Sun-vector estimation utilities for the MA4822 CubeSat project."""

from __future__ import annotations

import numpy as np

from sun_sensor import unit_vector


def estimate_sun_vector(
    sensor_normals: np.ndarray,
    readings: np.ndarray,
) -> np.ndarray:
    """Estimate the Sun direction from the Sun-sensor measurements.

    Each sensor contributes its outward normal vector weighted by its measured
    light intensity. For the six-face CubeSat arrangement, this reconstructs
    the Sun direction exactly in the ideal, noise-free case.

    Parameters
    ----------
    sensor_normals:
        Array with shape (N, 3), containing one unit normal per sensor.
    readings:
        Array with shape (N,), containing the measured sensor outputs.

    Returns
    -------
    np.ndarray
        Estimated unit Sun vector in the CubeSat body frame.
    """
    sensor_normals = np.asarray(sensor_normals, dtype=float)
    readings = np.asarray(readings, dtype=float)

    if sensor_normals.ndim != 2 or sensor_normals.shape[1] != 3:
        raise ValueError("sensor_normals must have shape (N, 3).")

    if readings.shape != (sensor_normals.shape[0],):
        raise ValueError("readings must contain one value per sensor.")

    if np.any(readings < 0):
        raise ValueError("Sensor readings must be non-negative.")

    # Weighted vector sum:
    #   s_raw = sum(I_i * n_i)
    #
    # For opposing sensors on +/-X, +/-Y, +/-Z, this is equivalent to:
    #   s_raw = [I_(+X)-I_(-X),
    #            I_(+Y)-I_(-Y),
    #            I_(+Z)-I_(-Z)]
    estimated_raw = sensor_normals.T @ readings

    if np.linalg.norm(estimated_raw) == 0:
        raise ValueError(
            "Sun direction cannot be estimated because all directional "
            "information is zero."
        )

    return unit_vector(estimated_raw)


def angular_error_degrees(
    true_vector: np.ndarray,
    estimated_vector: np.ndarray,
) -> float:
    """Return the angle between the true and estimated Sun vectors in degrees."""
    true_vector = unit_vector(true_vector)
    estimated_vector = unit_vector(estimated_vector)

    # Clip to avoid numerical errors such as 1.0000000002 inside arccos.
    cosine_angle = np.clip(
        np.dot(true_vector, estimated_vector),
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine_angle)))
