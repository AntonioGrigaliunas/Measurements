"""Final integrated benchmark for the self-aware CubeSat attitude system.

This script validates the final architecture:
- automatic Sun-sensor fault detection;
- two-hit persistent Sun-fault memory;
- adaptive Sun-vector reconstruction;
- magnetometer health gating;
- NORMAL / RECONFIGURED / SUN_SAFE / SAFE operating modes;
- fuzzy confidence output.

A smooth changing attitude is used so the same persistent fault is observed
across many different illumination geometries.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attitude_estimation import (
    attitude_error_degrees,
    rotation_matrix_from_euler,
)
from autonomous_attitude import (
    AutonomousAttitudeManager,
    autonomous_attitude_solution,
)
from magnetometer import Magnetometer
from sensor_parameters import (
    MAG_BIAS,
    MAG_FAULT_VECTOR,
    MAG_NOISE_STD,
    MAG_SCALE_FACTORS,
    SUN_GAINS,
    SUN_NOISE_STD,
    SUN_OFFSETS,
)
from sun_sensor import six_face_sensor_array, unit_vector


SUN_SENSOR_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def trajectory_rotation(sample: int, n_samples: int) -> np.ndarray:
    """Generate a smooth attitude trajectory that repeatedly changes lighting."""
    phase = 2.0 * np.pi * sample / n_samples

    roll = 25.0 + 35.0 * np.sin(2.0 * phase)
    pitch = -10.0 + 30.0 * np.sin(1.5 * phase + 0.6)
    yaw = 720.0 * sample / max(n_samples - 1, 1)

    return rotation_matrix_from_euler(
        roll_deg=roll,
        pitch_deg=pitch,
        yaw_deg=yaw,
    )


def make_models():
    sun_array = six_face_sensor_array(
        gains=SUN_GAINS,
        offsets=SUN_OFFSETS,
    )
    magnetometer = Magnetometer(
        scale_factors=MAG_SCALE_FACTORS,
        bias=MAG_BIAS,
    )
    return sun_array, magnetometer


def run_sequence(
    name: str,
    n_samples: int = 1200,
    fault_start: int = 300,
    seed: int = 42,
    failed_sun_sensor: int | None = None,
    magnetic_fault: bool = False,
) -> dict[str, object]:
    """Run one continuous scenario with persistent fault memory."""
    rng = np.random.default_rng(seed)

    sun_reference = unit_vector(np.array([1.0, 0.4, 0.2]))
    magnetic_reference = unit_vector(np.array([0.2, -0.3, 0.9]))

    sun_array, magnetometer = make_models()
    manager = AutonomousAttitudeManager(confirmations_required=2)

    modes = []
    reliabilities = []
    attitude_errors = []
    error_samples = []
    unsafe_outputs = 0
    full_outputs = 0
    latch_sample = None

    stateless_sun_fault_errors = []
    stateless_sun_fault_unsafe = 0
    stateless_sun_fault_full = 0

    for sample in range(n_samples):
        true_rotation = trajectory_rotation(sample, n_samples)
        true_sun_body = true_rotation @ sun_reference
        true_magnetic_body = true_rotation @ magnetic_reference

        sun_readings = sun_array.measure(
            true_sun_body,
            noise_std=SUN_NOISE_STD,
            rng=rng,
        )

        sun_fault_active = (
            failed_sun_sensor is not None
            and sample >= fault_start
        )
        if sun_fault_active:
            sun_readings[int(failed_sun_sensor)] = 0.0

        magnetic_measurement = magnetometer.measure(
            true_magnetic_body,
            noise_std=MAG_NOISE_STD,
            rng=rng,
        )

        if magnetic_fault and sample >= fault_start:
            magnetic_measurement = (
                magnetic_measurement + MAG_FAULT_VECTOR
            )

        solution = manager.update(
            sun_array.sensor_normals,
            sun_readings,
            magnetic_measurement,
            sun_reference,
            magnetic_reference,
        )

        modes.append(str(solution["mode"]))
        reliabilities.append(float(solution["reliability_score"]))

        if (
            latch_sample is None
            and solution["latched_failed_sun_sensor"] is not None
        ):
            latch_sample = sample

        if solution["rotation"] is not None:
            error = attitude_error_degrees(
                true_rotation,
                solution["rotation"],
            )
            attitude_errors.append(error)
            error_samples.append(sample)
            full_outputs += 1
            if error > 10.0:
                unsafe_outputs += 1

        # Direct stateless comparison for the persistent Sun-fault case.
        if sun_fault_active and not magnetic_fault:
            stateless = autonomous_attitude_solution(
                sun_array.sensor_normals,
                sun_readings,
                magnetic_measurement,
                sun_reference,
                magnetic_reference,
                gains=SUN_GAINS,
                offsets=SUN_OFFSETS,
            )
            if stateless["rotation"] is not None:
                stateless_error = attitude_error_degrees(
                    true_rotation,
                    stateless["rotation"],
                )
                stateless_sun_fault_errors.append(stateless_error)
                stateless_sun_fault_full += 1
                if stateless_error > 10.0:
                    stateless_sun_fault_unsafe += 1

    modes_counter = Counter(modes)
    attitude_errors_array = np.asarray(attitude_errors)
    reliabilities_array = np.asarray(reliabilities)

    post_fault_modes = Counter(modes[fault_start:])
    post_fault_reliability = reliabilities_array[fault_start:]

    result: dict[str, object] = {
        "name": name,
        "n_samples": n_samples,
        "fault_start": fault_start,
        "modes": modes_counter,
        "post_fault_modes": post_fault_modes,
        "reliability": reliabilities_array,
        "modes_series": modes,
        "attitude_errors": attitude_errors_array,
        "error_samples": np.asarray(error_samples),
        "full_attitude_fraction": full_outputs / n_samples,
        "unsafe_output_fraction": unsafe_outputs / n_samples,
        "mean_attitude_error": (
            float(np.mean(attitude_errors_array))
            if attitude_errors_array.size
            else np.nan
        ),
        "p95_attitude_error": (
            float(np.percentile(attitude_errors_array, 95))
            if attitude_errors_array.size
            else np.nan
        ),
        "mean_reliability": float(np.mean(reliabilities_array)),
        "post_fault_mean_reliability": float(
            np.mean(post_fault_reliability)
        ),
        "latch_sample": latch_sample,
    }

    if stateless_sun_fault_full:
        stateless_errors = np.asarray(stateless_sun_fault_errors)
        result["stateless_post_fault_mean_error"] = float(
            np.mean(stateless_errors)
        )
        result["stateless_post_fault_unsafe_fraction"] = (
            stateless_sun_fault_unsafe
            / (n_samples - fault_start)
        )
        result["persistent_post_fault_unsafe_fraction"] = (
            sum(
                1
                for sample, error in zip(
                    result["error_samples"],
                    result["attitude_errors"],
                )
                if sample >= fault_start and error > 10.0
            )
            / (n_samples - fault_start)
        )

    return result


def print_summary(results: list[dict[str, object]]) -> None:
    print("\nFinal integrated autonomous-system benchmark")
    print("--------------------------------------------")
    print(
        "Scenario       NORMAL  RECONFIG  SUN_SAFE  SAFE   Full att.  "
        "Mean err  P95 err  Unsafe>10  Mean rel."
    )
    print(
        "                 [%]      [%]       [%]     [%]      [%]      "
        "[deg]     [deg]      [%]       [%]"
    )

    for result in results:
        n_samples = int(result["n_samples"])
        modes = result["modes"]

        print(
            f"{str(result['name']):13s} "
            f"{100*modes['NORMAL']/n_samples:7.1f}  "
            f"{100*modes['RECONFIGURED']/n_samples:8.1f}  "
            f"{100*modes['SUN_SAFE']/n_samples:8.1f}  "
            f"{100*modes['SAFE']/n_samples:5.1f}  "
            f"{100*float(result['full_attitude_fraction']):8.1f}  "
            f"{float(result['mean_attitude_error']):8.3f}  "
            f"{float(result['p95_attitude_error']):7.3f}  "
            f"{100*float(result['unsafe_output_fraction']):8.2f}  "
            f"{float(result['mean_reliability']):8.2f}"
        )

    sun_result = next(
        (result for result in results if result["name"] == "Sun fault"),
        None,
    )

    if sun_result is not None:
        print("\nPersistent Sun-fault memory comparison")
        print("--------------------------------------")
        print(
            "Fault starts at sample "
            f"{sun_result['fault_start']}."
        )
        print(
            "Fault memory latched at sample: "
            f"{sun_result['latch_sample']}"
        )
        print(
            "Stateless post-fault mean full-attitude error: "
            f"{float(sun_result['stateless_post_fault_mean_error']):.3f} deg"
        )
        print(
            "Stateless post-fault unsafe >10 deg fraction: "
            f"{100*float(sun_result['stateless_post_fault_unsafe_fraction']):.2f}%"
        )
        print(
            "Persistent post-fault unsafe >10 deg fraction: "
            f"{100*float(sun_result['persistent_post_fault_unsafe_fraction']):.2f}%"
        )


def save_sun_memory_plot(
    result: dict[str, object],
    output_path: Path,
) -> None:
    """Plot fuzzy confidence and the persistent-fault latch event."""
    reliability = np.asarray(result["reliability"], dtype=float)
    samples = np.arange(len(reliability))
    fault_start = int(result["fault_start"])
    latch_sample = result["latch_sample"]

    plt.figure(figsize=(10, 5))
    plt.plot(samples, reliability, label="Fuzzy confidence")
    plt.axvline(
        fault_start,
        linestyle="--",
        label="Sun-sensor failure begins",
    )

    if latch_sample is not None:
        plt.axvline(
            int(latch_sample),
            linestyle=":",
            label="Fault memory latched",
        )

    plt.xlabel("Sample")
    plt.ylabel("Fuzzy confidence [%]")
    plt.ylim(0, 100)
    plt.title("Persistent Sun-fault memory and fuzzy confidence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_mode_timeline(
    result: dict[str, object],
    output_path: Path,
) -> None:
    """Plot the discrete operating mode selected over time."""
    mode_map = {
        "SAFE": 0,
        "SUN_SAFE": 1,
        "RECONFIGURED": 2,
        "NORMAL": 3,
    }
    values = np.array(
        [mode_map[mode] for mode in result["modes_series"]]
    )
    samples = np.arange(len(values))

    plt.figure(figsize=(10, 4.5))
    plt.step(samples, values, where="post")
    plt.axvline(
        int(result["fault_start"]),
        linestyle="--",
        label="Fault injection",
    )
    plt.yticks(
        [0, 1, 2, 3],
        ["SAFE", "SUN_SAFE", "RECONFIGURED", "NORMAL"],
    )
    plt.xlabel("Sample")
    plt.ylabel("Operating mode")
    plt.title(f"Autonomous mode timeline: {result['name']}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    results = [
        run_sequence("Healthy"),
        run_sequence(
            "Sun fault",
            failed_sun_sensor=0,
        ),
        run_sequence(
            "Mag fault",
            magnetic_fault=True,
        ),
        run_sequence(
            "Combined",
            failed_sun_sensor=0,
            magnetic_fault=True,
        ),
    ]

    print_summary(results)

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    sun_result = next(
        result for result in results
        if result["name"] == "Sun fault"
    )
    combined_result = next(
        result for result in results
        if result["name"] == "Combined"
    )

    memory_path = output_dir / "persistent_sun_fault_memory.png"
    timeline_path = output_dir / "final_combined_mode_timeline.png"

    save_sun_memory_plot(sun_result, memory_path)
    save_mode_timeline(combined_result, timeline_path)

    print("\nSaved figures:")
    print(memory_path)
    print(timeline_path)


if __name__ == "__main__":
    main()
