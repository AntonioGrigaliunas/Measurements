"""Simple three-axis magnetometer model for the MA4822 CubeSat project."""

from __future__ import annotations

import numpy as np

from sun_sensor import unit_vector


class Magnetometer:
    """Three-axis magnetometer with scale-factor, bias, and Gaussian noise."""

    def __init__(
        self,
        scale_factors: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> None:
        self.scale_factors = (
            np.ones(3)
            if scale_factors is None
            else np.asarray(scale_factors, dtype=float)
        )
        self.bias = (
            np.zeros(3)
            if bias is None
            else np.asarray(bias, dtype=float)
        )

        if self.scale_factors.shape != (3,):
            raise ValueError("scale_factors must contain three values.")

        if self.bias.shape != (3,):
            raise ValueError("bias must contain three values.")

    def measure(
        self,
        magnetic_field_body: np.ndarray,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return a simulated three-axis magnetic-field measurement."""
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative.")

        magnetic_field_body = unit_vector(magnetic_field_body)

        measurement = self.scale_factors * magnetic_field_body + self.bias

        if noise_std > 0:
            if rng is None:
                rng = np.random.default_rng()
            measurement = measurement + rng.normal(0.0, noise_std, 3)

        return measurement
