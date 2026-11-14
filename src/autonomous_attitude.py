"""Autonomous self-reconfiguring CubeSat attitude-determination logic.

Operating modes
---------------
NORMAL
    Sun array and magnetometer are accepted; full TRIAD attitude is available.
RECONFIGURED
    One Sun-sensor failure is detected or remembered and the Sun vector is
    reconstructed using the remaining channels; full TRIAD attitude remains
    available.
SUN_SAFE
    A usable Sun direction is available, but the magnetometer fails the
    magnetic-health checks. Full 3D attitude is intentionally withheld.
SAFE
    A trustworthy Sun direction cannot be produced.

The hard-fault logic protects and reconfigures the estimator. The fuzzy layer
has a separate job: it reports an interpretable confidence score.
"""

from __future__ import annotations

import numpy as np

from attitude_estimation import (
    angular_error_degrees,
    estimate_sun_vector,
    estimate_sun_vector_reconfigured,
    triad_attitude,
)
from fuzzy_reliability import fuzzy_reliability_score, reliability_label
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


def sun_reconstruction_residual(
    sensor_normals: np.ndarray,
    readings: np.ndarray,
    estimated_sun_body: np.ndarray,
    gains: np.ndarray,
    offsets: np.ndarray,
    healthy_mask: np.ndarray | None = None,
) -> float:
    """Return calibrated RMS Sun-sensor residual over usable channels."""
    ideal_prediction = np.maximum(
        0.0,
        np.asarray(sensor_normals, dtype=float) @ estimated_sun_body,
    )
    predicted = np.maximum(
        0.0,
        np.asarray(gains, dtype=float) * ideal_prediction
        + np.asarray(offsets, dtype=float),
    )

    residuals = np.asarray(readings, dtype=float) - predicted

    if healthy_mask is not None:
        healthy_mask = np.asarray(healthy_mask, dtype=bool)
        residuals = residuals[healthy_mask]

    if residuals.size == 0:
        return float("inf")

    return float(np.sqrt(np.mean(residuals**2)))


def _reliability_output(
    sensor_normals: np.ndarray,
    sun_readings: np.ndarray,
    magnetic_measurement: np.ndarray,
    estimated_sun_body: np.ndarray,
    sun_reference: np.ndarray,
    magnetic_reference: np.ndarray,
    gains: np.ndarray,
    offsets: np.ndarray,
    failed_sensor: int | None,
) -> dict[str, object]:
    """Calculate the fuzzy confidence score for the current sensor solution."""
    healthy_mask = np.ones(len(sun_readings), dtype=bool)
    if failed_sensor is not None:
        healthy_mask[int(failed_sensor)] = False

    magnetic_health = assess_magnetometer_health(
        estimated_sun_body,
        magnetic_measurement,
        sun_reference,
        magnetic_reference,
    )

    residual = sun_reconstruction_residual(
        sensor_normals,
        sun_readings,
        estimated_sun_body,
        gains,
        offsets,
        healthy_mask=healthy_mask,
    )

    sun_health_fraction = float(np.mean(healthy_mask))

    score, possibilities = fuzzy_reliability_score(
        sun_health_fraction,
        float(magnetic_health["consistency_error_deg"]),
        float(magnetic_health["norm_error"]),
        residual,
    )

    return {
        "score": float(score),
        "label": reliability_label(score),
        "possibilities": possibilities,
        "sun_health_fraction": sun_health_fraction,
        "sun_residual": residual,
        "magnetometer_health": magnetic_health,
    }


def autonomous_attitude_solution(
    sensor_normals: np.ndarray,
    sun_readings: np.ndarray,
    magnetic_measurement: np.ndarray,
    sun_reference: np.ndarray,
    magnetic_reference: np.ndarray,
    gains: np.ndarray = SUN_GAINS,
    offsets: np.ndarray = SUN_OFFSETS,
    remembered_failed_sun_sensor: int | None = None,
) -> dict[str, object]:
    """Return one autonomous attitude solution.

    remembered_failed_sun_sensor lets a stateful manager force continued
    reconfiguration after a previously confirmed hard failure.
    """
    detection = detect_failed_sun_sensor(
        sun_readings,
        magnetic_measurement,
        sun_reference,
        magnetic_reference,
        gains,
        offsets,
    )
    detected_sensor = detection["failed_sensor"]

    if remembered_failed_sun_sensor is not None:
        active_failed_sensor = int(remembered_failed_sun_sensor)
        failure_source = "memory"
    elif detected_sensor is not None:
        active_failed_sensor = int(detected_sensor)
        failure_source = "current_detection"
    else:
        active_failed_sensor = None
        failure_source = "none"

    try:
        if active_failed_sensor is None:
            estimated_sun_body = estimate_sun_vector(
                sensor_normals,
                sun_readings,
            )
            sun_reconfigured = False
        else:
            healthy_mask = np.ones(len(sun_readings), dtype=bool)
            healthy_mask[active_failed_sensor] = False
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
            "detected_failed_sun_sensor": detected_sensor,
            "active_failed_sun_sensor": active_failed_sensor,
            "failure_source": failure_source,
            "sun_fault_detection": detection,
            "magnetometer_health": None,
            "reliability_score": 0.0,
            "reliability_label": "Very Low",
            "reliability_details": None,
        }

    reliability = _reliability_output(
        sensor_normals,
        sun_readings,
        magnetic_measurement,
        estimated_sun_body,
        sun_reference,
        magnetic_reference,
        gains,
        offsets,
        active_failed_sensor,
    )
    magnetic_health = reliability["magnetometer_health"]

    if not bool(magnetic_health["healthy"]):
        return {
            "mode": "SUN_SAFE",
            "rotation": None,
            "sun_body": estimated_sun_body,
            "magnetic_body": unit_vector(magnetic_measurement),
            "detected_failed_sun_sensor": detected_sensor,
            "active_failed_sun_sensor": active_failed_sensor,
            "failure_source": failure_source,
            "sun_fault_detection": detection,
            "magnetometer_health": magnetic_health,
            "reliability_score": reliability["score"],
            "reliability_label": reliability["label"],
            "reliability_details": reliability,
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
            "detected_failed_sun_sensor": detected_sensor,
            "active_failed_sun_sensor": active_failed_sensor,
            "failure_source": failure_source,
            "sun_fault_detection": detection,
            "magnetometer_health": magnetic_health,
            "reliability_score": reliability["score"],
            "reliability_label": reliability["label"],
            "reliability_details": reliability,
        }

    return {
        "mode": "RECONFIGURED" if sun_reconfigured else "NORMAL",
        "rotation": rotation,
        "sun_body": estimated_sun_body,
        "magnetic_body": estimated_magnetic_body,
        "detected_failed_sun_sensor": detected_sensor,
        "active_failed_sun_sensor": active_failed_sensor,
        "failure_source": failure_source,
        "sun_fault_detection": detection,
        "magnetometer_health": magnetic_health,
        "reliability_score": reliability["score"],
        "reliability_label": reliability["label"],
        "reliability_details": reliability,
    }


class AutonomousAttitudeManager:
    """Stateful attitude manager with persistent Sun-sensor fault memory.

    A current detector hit can trigger immediate reconfiguration for that
    sample. Persistent memory is more conservative: the same sensor must be
    detected several times within a rolling confirmation window. This prevents
    rare isolated false alarms from accumulating indefinitely during long
    missions.

    Once a hard fault is latched, it remains latched until reset().
    """

    def __init__(
        self,
        confirmations_required: int = 3,
        confirmation_window: int = 20,
        gains: np.ndarray = SUN_GAINS,
        offsets: np.ndarray = SUN_OFFSETS,
    ) -> None:
        if confirmations_required < 1:
            raise ValueError("confirmations_required must be at least 1.")
        if confirmation_window < 1:
            raise ValueError("confirmation_window must be at least 1.")

        self.confirmations_required = int(confirmations_required)
        self.confirmation_window = int(confirmation_window)
        self.gains = np.asarray(gains, dtype=float)
        self.offsets = np.asarray(offsets, dtype=float)
        self.reset()

    def reset(self) -> None:
        """Clear all remembered fault state."""
        self.latched_failed_sun_sensor: int | None = None
        self._sample_index = -1
        self._detection_history: list[tuple[int, int]] = []

    def _update_fault_memory(self, detected_sensor: int | None) -> None:
        self._sample_index += 1

        # Keep only recent evidence. Old isolated false alarms expire.
        oldest_allowed = self._sample_index - self.confirmation_window + 1
        self._detection_history = [
            (sample, sensor)
            for sample, sensor in self._detection_history
            if sample >= oldest_allowed
        ]

        if self.latched_failed_sun_sensor is not None:
            return

        if detected_sensor is not None:
            self._detection_history.append(
                (self._sample_index, int(detected_sensor))
            )

        counts = np.zeros(6, dtype=int)
        for _, sensor in self._detection_history:
            if 0 <= sensor < 6:
                counts[sensor] += 1

        if np.max(counts) >= self.confirmations_required:
            self.latched_failed_sun_sensor = int(np.argmax(counts))

    def update(
        self,
        sensor_normals: np.ndarray,
        sun_readings: np.ndarray,
        magnetic_measurement: np.ndarray,
        sun_reference: np.ndarray,
        magnetic_reference: np.ndarray,
    ) -> dict[str, object]:
        """Process one sensor sample and update persistent fault memory."""
        detection = detect_failed_sun_sensor(
            sun_readings,
            magnetic_measurement,
            sun_reference,
            magnetic_reference,
            self.gains,
            self.offsets,
        )
        detected_sensor = detection["failed_sensor"]
        self._update_fault_memory(
            None if detected_sensor is None else int(detected_sensor)
        )

        solution = autonomous_attitude_solution(
            sensor_normals,
            sun_readings,
            magnetic_measurement,
            sun_reference,
            magnetic_reference,
            gains=self.gains,
            offsets=self.offsets,
            remembered_failed_sun_sensor=self.latched_failed_sun_sensor,
        )

        recent_counts = np.zeros(6, dtype=int)
        for _, sensor in self._detection_history:
            if 0 <= sensor < 6:
                recent_counts[sensor] += 1

        solution["latched_failed_sun_sensor"] = (
            self.latched_failed_sun_sensor
        )
        solution["fault_evidence_counts"] = recent_counts.copy()
        solution["fault_confirmation_window"] = self.confirmation_window
        solution["fault_confirmations_required"] = (
            self.confirmations_required
        )

        return solution

