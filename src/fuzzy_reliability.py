"""Fuzzy reliability assessment for the MA4822 CubeSat project.

The structure follows the course approach:
1. Crisp sensor-quality indicators -> fuzzy beliefs.
2. IF-THEN fuzzy rules -> fuzzy reliability decisions.
3. Weighted-sum defuzzification -> crisp reliability score.
"""

from __future__ import annotations


def triangular(x: float, a: float, b: float, c: float) -> float:
    """Triangular possibility function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoidal(
    x: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float:
    """Trapezoidal possibility function."""
    if x <= a:
        return 1.0 if a == b else 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    if x < d:
        return (d - x) / (d - c)
    return 1.0 if c == d and x >= d else 0.0


def fuzzify_inputs(
    sun_health_fraction: float,
    vector_consistency_error_deg: float,
    magnetometer_norm_error: float,
    sun_reconstruction_residual: float,
) -> dict[str, dict[str, float]]:
    """Convert crisp quality indicators into fuzzy beliefs."""
    health = {
        "poor": trapezoidal(sun_health_fraction, 0.0, 0.0, 0.55, 0.75),
        "degraded": triangular(sun_health_fraction, 0.65, 5.0 / 6.0, 1.0),
        "good": trapezoidal(sun_health_fraction, 5.0 / 6.0, 1.0, 1.0, 1.0),
    }

    consistency = {
        "good": trapezoidal(vector_consistency_error_deg, 0.0, 0.0, 1.0, 3.0),
        "moderate": triangular(vector_consistency_error_deg, 1.0, 5.0, 10.0),
        "poor": trapezoidal(vector_consistency_error_deg, 6.0, 12.0, 30.0, 30.0),
    }

    magnetic = {
        "good": trapezoidal(magnetometer_norm_error, 0.0, 0.0, 0.02, 0.05),
        "moderate": triangular(magnetometer_norm_error, 0.02, 0.07, 0.15),
        "poor": trapezoidal(magnetometer_norm_error, 0.08, 0.15, 0.50, 0.50),
    }

    sun_residual = {
        "good": trapezoidal(sun_reconstruction_residual, 0.0, 0.0, 0.015, 0.035),
        "moderate": triangular(sun_reconstruction_residual, 0.02, 0.055, 0.10),
        "poor": trapezoidal(sun_reconstruction_residual, 0.06, 0.12, 0.50, 0.50),
    }

    return {
        "sun_health": health,
        "consistency": consistency,
        "magnetic": magnetic,
        "sun_residual": sun_residual,
    }


def fuzzy_reliability_score(
    sun_health_fraction: float,
    vector_consistency_error_deg: float,
    magnetometer_norm_error: float,
    sun_reconstruction_residual: float,
) -> tuple[float, dict[str, float]]:
    """Return a 0-100 reliability score and fuzzy output possibilities."""
    beliefs = fuzzify_inputs(
        sun_health_fraction,
        vector_consistency_error_deg,
        magnetometer_norm_error,
        sun_reconstruction_residual,
    )

    h = beliefs["sun_health"]
    c = beliefs["consistency"]
    m = beliefs["magnetic"]
    s = beliefs["sun_residual"]

    # AND is implemented with MIN, matching the MA4822 course slides.
    high_rules = [
        min(h["good"], c["good"], m["good"], s["good"]),
    ]

    medium_rules = [
        min(h["degraded"], c["good"], m["good"], s["good"]),
        min(h["good"], c["moderate"], m["good"], s["good"]),
        min(h["good"], c["good"], m["moderate"], s["good"]),
        min(h["good"], c["good"], m["good"], s["moderate"]),
        min(h["degraded"], c["good"], m["good"], s["moderate"]),
        min(h["degraded"], c["moderate"], m["good"], s["good"]),
        min(h["degraded"], c["good"], m["moderate"], s["good"]),
    ]

    # Any clearly poor sensor-quality indicator should lower confidence.
    low_rules = [
        h["poor"],
        c["poor"],
        m["poor"],
        s["poor"],
        min(h["degraded"], s["moderate"]),
        min(c["moderate"], m["moderate"]),
        min(c["moderate"], s["moderate"]),
    ]

    possibilities = {
        "low": max(low_rules),
        "medium": max(medium_rules),
        "high": max(high_rules),
    }

    # Weighted-sum defuzzification using representative crisp values.
    output_values = {
        "low": 20.0,
        "medium": 60.0,
        "high": 90.0,
    }

    denominator = sum(possibilities.values())

    if denominator == 0.0:
        return 50.0, possibilities

    score = sum(
        output_values[name] * possibility
        for name, possibility in possibilities.items()
    ) / denominator

    return float(score), possibilities


def reliability_label(score: float) -> str:
    """Convert the crisp reliability score into a readable label."""
    if score < 40.0:
        return "Low"
    if score < 75.0:
        return "Medium"
    return "High"
