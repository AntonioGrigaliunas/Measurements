"""First executable check of the CubeSat Sun-sensor model."""

import numpy as np

from sun_sensor import six_face_sensor_array, unit_vector


def main() -> None:
    sensor_array = six_face_sensor_array()

    # Example true Sun direction in the CubeSat body frame.
    true_sun_vector = unit_vector(np.array([1.0, 1.0, 0.5]))

    readings = sensor_array.measure(true_sun_vector)

    face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

    print("True Sun vector:")
    print(np.round(true_sun_vector, 4))
    print("\nSensor readings:")

    for face, reading in zip(face_names, readings):
        print(f"{face}: {reading:.4f}")


if __name__ == "__main__":
    main()
