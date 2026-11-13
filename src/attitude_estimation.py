"""Attitude-estimation utilities for the MA4822 CubeSat project."""

from __future__ import annotations

import numpy as np

from sun_sensor import unit_vector


def estimate_sun_vector(
    sensor_normals: np.ndarray,
    readings: np.ndarray,
) -> np.ndarray:
    """Estimate the Sun direction from the Sun-sensor measurements."""
    sensor_normals = np.asarray(sensor_normals, dtype=float)
    readings = np.asarray(readings, dtype=float)

    if sensor_normals.ndim != 2 or sensor_normals.shape[1] != 3:
        raise ValueError("sensor_normals must have shape (N, 3).")

    if readings.shape != (sensor_normals.shape[0],):
        raise ValueError("readings must contain one value per sensor.")

    if np.any(readings < 0):
        raise ValueError("Sensor readings must be non-negative.")

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
    """Return the angle between two vectors in degrees."""
    true_vector = unit_vector(true_vector)
    estimated_vector = unit_vector(estimated_vector)

    cosine_angle = np.clip(
        np.dot(true_vector, estimated_vector),
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine_angle)))


def triad_attitude(
    sun_reference: np.ndarray,
    magnetic_reference: np.ndarray,
    sun_body: np.ndarray,
    magnetic_body: np.ndarray,
) -> np.ndarray:
    """Estimate the body-from-reference rotation matrix using TRIAD.

    The two reference vectors are known in the global/reference frame, while
    the two measured vectors are expressed in the CubeSat body frame.
    """
    s_r = unit_vector(sun_reference)
    m_r = unit_vector(magnetic_reference)
    s_b = unit_vector(sun_body)
    m_b = unit_vector(magnetic_body)

    cross_r = np.cross(s_r, m_r)
    cross_b = np.cross(s_b, m_b)

    if np.linalg.norm(cross_r) < 1e-8:
        raise ValueError("Reference vectors are too close to parallel for TRIAD.")

    if np.linalg.norm(cross_b) < 1e-8:
        raise ValueError("Measured vectors are too close to parallel for TRIAD.")

    t1_r = s_r
    t2_r = unit_vector(cross_r)
    t3_r = np.cross(t1_r, t2_r)

    t1_b = s_b
    t2_b = unit_vector(cross_b)
    t3_b = np.cross(t1_b, t2_b)

    reference_triad = np.column_stack((t1_r, t2_r, t3_r))
    body_triad = np.column_stack((t1_b, t2_b, t3_b))

    # Maps vectors from the reference frame into the CubeSat body frame.
    return body_triad @ reference_triad.T


def attitude_error_degrees(
    true_rotation: np.ndarray,
    estimated_rotation: np.ndarray,
) -> float:
    """Return the principal rotation-angle error between two attitudes."""
    true_rotation = np.asarray(true_rotation, dtype=float)
    estimated_rotation = np.asarray(estimated_rotation, dtype=float)

    if true_rotation.shape != (3, 3) or estimated_rotation.shape != (3, 3):
        raise ValueError("Rotation matrices must have shape (3, 3).")

    error_rotation = estimated_rotation @ true_rotation.T
    cosine_angle = np.clip(
        (np.trace(error_rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine_angle)))


def rotation_matrix_from_euler(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Create a body-from-reference rotation matrix from XYZ attitude angles."""
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    ry = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rz @ ry @ rx
