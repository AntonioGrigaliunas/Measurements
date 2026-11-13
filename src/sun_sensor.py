"""Simple Sun-sensor model for the MA4822 CubeSat project.

The model assumes each light sensor follows an ideal cosine response:
    measurement = max(0, n_i dot s)

where n_i is the outward unit normal of sensor i and s is the unit Sun vector
expressed in the CubeSat body frame.

Gain, offset, and Gaussian noise terms are included so that the project can
study calibration errors and measurement uncertainty.
"""

from __future__ import annotations

import numpy as np


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a normalized copy of a 3D vector."""
    vector = np.asarray(vector, dtype=float)

    if vector.shape != (3,):
        raise ValueError("Expected a 3-element vector.")

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("A zero vector cannot be normalized.")

    return vector / norm


class SunSensorArray:
    """Array of Sun sensors with known body-frame orientations."""

    def __init__(
        self,
        sensor_normals: np.ndarray,
        gains: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
    ) -> None:
        sensor_normals = np.asarray(sensor_normals, dtype=float)

        if sensor_normals.ndim != 2 or sensor_normals.shape[1] != 3:
            raise ValueError("sensor_normals must have shape (N, 3).")

        norms = np.linalg.norm(sensor_normals, axis=1)
        if np.any(norms == 0):
            raise ValueError("Sensor normals cannot contain zero vectors.")

        self.sensor_normals = sensor_normals / norms[:, None]
        self.n_sensors = self.sensor_normals.shape[0]

        self.gains = (
            np.ones(self.n_sensors)
            if gains is None
            else np.asarray(gains, dtype=float)
        )
        self.offsets = (
            np.zeros(self.n_sensors)
            if offsets is None
            else np.asarray(offsets, dtype=float)
        )

        if self.gains.shape != (self.n_sensors,):
            raise ValueError("gains must have one value per sensor.")

        if self.offsets.shape != (self.n_sensors,):
            raise ValueError("offsets must have one value per sensor.")

    def measure(
        self,
        sun_vector: np.ndarray,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return the simulated reading of every sensor.

        Parameters
        ----------
        sun_vector:
            Sun direction in the CubeSat body frame.
        noise_std:
            Standard deviation of additive Gaussian measurement noise.
        rng:
            Optional NumPy random generator for reproducible simulations.
        """
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative.")

        sun_vector = unit_vector(sun_vector)

        # Ideal cosine response. Sensors facing away from the Sun read zero.
        ideal_response = np.maximum(0.0, self.sensor_normals @ sun_vector)

        # Static calibration mismatch.
        readings = self.gains * ideal_response + self.offsets

        # Random measurement noise.
        if noise_std > 0:
            if rng is None:
                rng = np.random.default_rng()
            readings = readings + rng.normal(0.0, noise_std, self.n_sensors)

        # A light sensor cannot produce a physically meaningful negative signal.
        return np.maximum(0.0, readings)


def six_face_sensor_array(
    gains: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
) -> SunSensorArray:
    """Create one Sun sensor on each face of a cubic CubeSat."""
    normals = np.array(
        [
            [1.0, 0.0, 0.0],   # +X
            [-1.0, 0.0, 0.0],  # -X
            [0.0, 1.0, 0.0],   # +Y
            [0.0, -1.0, 0.0],  # -Y
            [0.0, 0.0, 1.0],   # +Z
            [0.0, 0.0, -1.0],  # -Z
        ]
    )
    return SunSensorArray(normals, gains=gains, offsets=offsets)
