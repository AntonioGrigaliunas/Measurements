"""Run the first complete Sun-sensor estimation example."""

import numpy as np

from attitude_estimation import angular_error_degrees, estimate_sun_vector
from sun_sensor import six_face_sensor_array, unit_vector


def main() -> None:
    sensor_array = six_face_sensor_array()

    # Ground truth used only by the simulation.
    # The real spacecraft would not know this vector directly.
    true_sun_vector = unit_vector(np.array([1.0, 1.0, 0.5]))

    # Simulate the six sensor measurements.
    readings = sensor_array.measure(true_sun_vector)

    # Reconstruct the Sun direction using only the sensor readings.
    estimated_sun_vector = estimate_sun_vector(
        sensor_array.sensor_normals,
        readings,
    )

    # Because this is a simulation, we can compare the estimate with ground truth.
    error_deg = angular_error_degrees(
        true_sun_vector,
        estimated_sun_vector,
    )

    face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

    print("True Sun vector:")
    print(np.round(true_sun_vector, 4))

    print("\nSensor readings:")
    for face, reading in zip(face_names, readings):
        print(f"{face}: {reading:.4f}")

    print("\nEstimated Sun vector:")
    print(np.round(estimated_sun_vector, 4))

    print(f"\nAngular estimation error: {error_deg:.6f} deg")


if __name__ == "__main__":
    main()
