"""Generate the final figures used in the MA4822 report.

The estimator and fault logic are treated as frozen. This script only reruns
the existing validation analyses and creates publication-ready figures in
report/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from automatic_reconfiguration_analysis import (
    SUN_SENSOR_NAMES,
    healthy_false_alarm_rate,
    run_failed_sensor_case,
)
from final_system_analysis import run_sequence


OUTPUT_DIR = Path("report") / "figures"


def save_automatic_reconfiguration_comparison() -> None:
    """Compare baseline, automatic, and perfect-known-fault reconstruction."""
    results = [
        run_failed_sensor_case(index)
        for index in range(len(SUN_SENSOR_NAMES))
    ]

    names = [str(result["sensor"]) for result in results]
    standard = [float(result["standard_mean"]) for result in results]
    automatic = [float(result["automatic_mean"]) for result in results]
    oracle = [float(result["oracle_mean"]) for result in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - width, standard, width, label="Standard estimator")
    ax.bar(x, automatic, width, label="Automatic reconfiguration")
    ax.bar(x + width, oracle, width, label="Known-fault benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("Failed Sun-sensor face")
    ax.set_ylabel("Mean attitude error [deg]")
    ax.set_title("Automatic recovery from a failed Sun sensor")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final_sun_reconfiguration.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        "Healthy instantaneous false-alarm rate: "
        f"{100.0 * healthy_false_alarm_rate():.2f}%"
    )


def final_sequences() -> list[dict[str, object]]:
    """Run the four frozen final-system scenarios."""
    return [
        run_sequence("Healthy"),
        run_sequence("Sun fault", failed_sun_sensor=0),
        run_sequence("Mag fault", magnetic_fault=True),
        run_sequence(
            "Combined",
            failed_sun_sensor=0,
            magnetic_fault=True,
        ),
    ]


def save_post_fault_modes(results: list[dict[str, object]]) -> None:
    """Plot post-fault operating-mode selection as 100% stacked bars."""
    names = [str(result["name"]) for result in results]
    mode_names = ["NORMAL", "RECONFIGURED", "SUN_SAFE", "SAFE"]

    values_by_mode: dict[str, list[float]] = {
        mode: [] for mode in mode_names
    }

    for result in results:
        n_post = int(result["n_samples"]) - int(result["fault_start"])
        modes = result["post_fault_modes"]
        for mode in mode_names:
            values_by_mode[mode].append(
                100.0 * modes[mode] / n_post
            )

    x = np.arange(len(names))
    bottom = np.zeros(len(names))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for mode in mode_names:
        values = np.asarray(values_by_mode[mode], dtype=float)
        ax.bar(x, values, bottom=bottom, label=mode)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Fraction of post-fault samples [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Autonomous operating-mode selection")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final_post_fault_modes.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_post_fault_reliability(results: list[dict[str, object]]) -> None:
    """Plot mean fuzzy confidence after fault injection."""
    names = [str(result["name"]) for result in results]
    reliability = [
        float(result["post_fault_mean_reliability"])
        for result in results
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(names, reliability)
    ax.set_ylabel("Mean fuzzy confidence [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Fuzzy confidence after fault injection")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, reliability):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final_fuzzy_confidence.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_persistent_memory_comparison(
    results: list[dict[str, object]],
) -> None:
    """Show how temporal fault memory reduces unsafe Sun-fault outputs."""
    sun_result = next(
        result for result in results
        if result["name"] == "Sun fault"
    )

    stateless = (
        100.0
        * float(sun_result["stateless_post_fault_unsafe_fraction"])
    )
    persistent = (
        100.0
        * float(sun_result["persistent_post_fault_unsafe_fraction"])
    )

    labels = ["Stateless", "Persistent memory"]
    values = [stateless, persistent]

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Unsafe outputs >10 deg [% of post-fault samples]")
    ax.set_title("Benefit of persistent Sun-sensor fault memory")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.12,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "final_persistent_memory_benefit.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_automatic_reconfiguration_comparison()
    results = final_sequences()
    save_post_fault_modes(results)
    save_post_fault_reliability(results)
    save_persistent_memory_comparison(results)

    print("\nFinal report figures saved in:")
    print(OUTPUT_DIR)
    for path in sorted(OUTPUT_DIR.glob("final_*.png")):
        print(path)


if __name__ == "__main__":
    main()
