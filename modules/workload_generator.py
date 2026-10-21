
import numpy as np
import pandas as pd


def generate_workload_data(num_samples=5000, seed=42):

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=num_samples,
        freq="min"
    )

    minutes = np.arange(num_samples)

    # Normal daily workload pattern
    daily_pattern = (
        20 + 15 * np.sin(
            2 * np.pi * minutes / 1440 - np.pi / 2
        )
    )

    # Short-term workload fluctuations
    short_term_pattern = (
        7 * np.sin(2 * np.pi * minutes / 180)
    )

    # Distribute workload surges across the entire dataset
    surges = np.zeros(num_samples)

    number_of_surges = max(1, num_samples // 500)

    surge_positions = np.linspace(
        100,
        num_samples - 100,
        number_of_surges,
        dtype=int
    )

    for position in surge_positions:

        start = int(
            np.clip(
                position + rng.integers(-40, 41),
                0,
                num_samples - 1
            )
        )

        ramp_duration = int(rng.integers(8, 16))
        peak_duration = int(rng.integers(10, 26))
        amplitude = rng.uniform(35, 50)

        for offset in range(
            ramp_duration + peak_duration + ramp_duration
        ):

            current_position = start + offset

            if current_position >= num_samples:
                break

            if offset < ramp_duration:
                intensity = (offset + 1) / ramp_duration

            elif offset < ramp_duration + peak_duration:
                intensity = 1.0

            else:
                intensity = (
                    1 - (
                        offset - ramp_duration - peak_duration
                    ) / ramp_duration
                )

            surges[current_position] += (
                amplitude * max(0, intensity)
            )

    # Generate CPU utilization
    noise = rng.normal(0, 2.5, num_samples)

    cpu = (
        daily_pattern
        + short_term_pattern
        + surges
        + noise
    )

    cpu = np.clip(cpu, 5, 100)

    # Generate active users based on workload
    active_users = (
        100 + cpu * 18
        + rng.normal(0, 35, num_samples)
    )

    active_users = np.clip(
        active_users, 10, 5000
    ).astype(int)

    # Generate requests per second
    requests_per_second = (
        active_users
        * rng.uniform(0.15, 0.35, num_samples)
    )

    requests_per_second = np.clip(
        requests_per_second, 1, 2000
    )

    # Generate memory utilization
    memory = (
        15 + cpu * 0.65
        + rng.normal(0, 3, num_samples)
    )

    memory = np.clip(memory, 10, 100)

    # Generate network traffic in MB
    network_traffic = (
        requests_per_second
        * rng.uniform(0.02, 0.08, num_samples)
    )

    network_traffic = np.clip(
        network_traffic, 0.1, 500
    )

    # Target: CPU utilization five minutes ahead
    future_cpu = np.roll(cpu, -5)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_usage": cpu,
        "memory_usage": memory,
        "active_users": active_users,
        "requests_per_second": requests_per_second,
        "network_traffic_mb": network_traffic,
        "future_cpu_usage": future_cpu
    })

    # Remove the last five records because their future
    # CPU values are not available
    return data.iloc[:-5].copy()
