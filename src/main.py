"""Demonstrate full 3D CubeSat attitude estimation using two sensors."""

import numpy as np

from attitude_estimation import (
    angular_error_degrees,
    attitude_error_degrees,
    estimate_sun_vector,
    rotation_matrix_from_euler,
    triad_attitude,
)
from magnetometer import Magnetometer
from sensor_parameters import (
    MAG_BIAS,
    MAG_NOISE_STD,
    MAG_SCALE_FACTORS,
    SUN_GAINS,
    SUN_NOISE_STD,
    SUN_OFFSETS,
)
from sun_sensor import six_face_sensor_array, unit_vector


def main() -> None:
    rng = np.random.default_rng(42)

    # Known environmental reference vectors.
    sun_reference = unit_vector(np.array([1.0, 0.4, 0.2]))
    magnetic_reference = unit_vector(np.array([0.2, -0.3, 0.9]))

    # Ground-truth CubeSat attitude, known only inside the simulation.
    true_rotation = rotation_matrix_from_euler(
        roll_deg=20.0,
        pitch_deg=-15.0,
        yaw_deg=35.0,
    )

    # Convert the known reference vectors into the CubeSat body frame.
    true_sun_body = true_rotation @ sun_reference
    true_magnetic_body = true_rotation @ magnetic_reference

    # Sun-sensor model with centralized hardware-based assumptions.
    sun_array = six_face_sensor_array(
        gains=SUN_GAINS,
        offsets=SUN_OFFSETS,
    )

    sun_readings = sun_array.measure(
        true_sun_body,
        noise_std=SUN_NOISE_STD,
        rng=rng,
    )

    estimated_sun_body = estimate_sun_vector(
        sun_array.sensor_normals,
        sun_readings,
    )

    # RM3100-class three-axis magnetometer model.
    magnetometer = Magnetometer(
        scale_factors=MAG_SCALE_FACTORS,
        bias=MAG_BIAS,
    )

    magnetic_measurement = magnetometer.measure(
        true_magnetic_body,
        noise_std=MAG_NOISE_STD,
        rng=rng,
    )

    estimated_magnetic_body = unit_vector(magnetic_measurement)

    # Fuse the two independent vector observations using TRIAD.
    estimated_rotation = triad_attitude(
        sun_reference,
        magnetic_reference,
        estimated_sun_body,
        estimated_magnetic_body,
    )

    sun_error = angular_error_degrees(
        true_sun_body,
        estimated_sun_body,
    )

    magnetic_error = angular_error_degrees(
        true_magnetic_body,
        estimated_magnetic_body,
    )

    attitude_error = attitude_error_degrees(
        true_rotation,
        estimated_rotation,
    )

    print("Reference-frame Sun vector:")
    print(np.round(sun_reference, 4))

    print("\nReference-frame magnetic vector:")
    print(np.round(magnetic_reference, 4))

    print("\nTrue body-frame Sun vector:")
    print(np.round(true_sun_body, 4))

    print("\nEstimated body-frame Sun vector:")
    print(np.round(estimated_sun_body, 4))
    print(f"Sun-vector error: {sun_error:.4f} deg")

    print("\nTrue body-frame magnetic vector:")
    print(np.round(true_magnetic_body, 4))

    print("\nMeasured/estimated magnetic direction:")
    print(np.round(estimated_magnetic_body, 4))
    print(f"Magnetic-vector error: {magnetic_error:.4f} deg")

    print("\nTrue body-from-reference rotation matrix:")
    print(np.round(true_rotation, 4))

    print("\nEstimated body-from-reference rotation matrix:")
    print(np.round(estimated_rotation, 4))

    print(f"\nFull attitude error: {attitude_error:.4f} deg")


if __name__ == "__main__":
    main()
