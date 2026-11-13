"""Compare ideal and imperfect CubeSat Sun-sensor measurements."""

import numpy as np

from attitude_estimation import angular_error_degrees, estimate_sun_vector
from sun_sensor import six_face_sensor_array, unit_vector


def evaluate_case(
    name: str,
    sensor_array,
    true_sun_vector: np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
    """Simulate one sensor case and print its Sun-vector estimate."""
    readings = sensor_array.measure(
        true_sun_vector,
        noise_std=noise_std,
        rng=rng,
    )

    estimated_sun_vector = estimate_sun_vector(
        sensor_array.sensor_normals,
        readings,
    )

    error_deg = angular_error_degrees(
        true_sun_vector,
        estimated_sun_vector,
    )

    face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

    print(f"\n--- {name} ---")
    print("Sensor readings:")
    for face, reading in zip(face_names, readings):
        print(f"{face}: {reading:.4f}")

    print("Estimated Sun vector:")
    print(np.round(estimated_sun_vector, 4))

    print(f"Angular estimation error: {error_deg:.4f} deg")


def main() -> None:
    # Ground truth used only by the simulation.
    true_sun_vector = unit_vector(np.array([1.0, 1.0, 0.5]))

    print("True Sun vector:")
    print(np.round(true_sun_vector, 4))

    # Case 1: perfect sensors.
    ideal_array = six_face_sensor_array()

    evaluate_case(
        "Ideal sensors",
        ideal_array,
        true_sun_vector,
    )

    # Case 2: realistic imperfections.
    #
    # Gain values represent small sensitivity differences between sensors.
    # Offset values represent small zero-reading biases.
    gains = np.array([1.03, 0.98, 0.96, 1.02, 1.05, 0.97])
    offsets = np.array([0.010, 0.005, 0.000, 0.008, 0.012, 0.004])

    imperfect_array = six_face_sensor_array(
        gains=gains,
        offsets=offsets,
    )

    # Fixed seed makes the example reproducible.
    rng = np.random.default_rng(42)

    evaluate_case(
        "Imperfect sensors",
        imperfect_array,
        true_sun_vector,
        noise_std=0.02,
        rng=rng,
    )


if __name__ == "__main__":
    main()
