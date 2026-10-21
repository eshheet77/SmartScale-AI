
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# LIVE WORKLOAD GENERATOR
# ---------------------------------------------------------

def generate_workload(step, seed=42):
    """
    Generates one simulated cloud workload reading.

    The workload follows a repeating pattern:
    Normal -> Increasing -> High Traffic -> Cooldown.

    Every generated metric is correlated with the workload
    intensity to create realistic demonstration data.
    """

    rng = np.random.default_rng(seed + step)

    cycle_position = step % 20

    # Determine workload intensity based on simulation step
    if cycle_position < 5:
        # Normal traffic
        intensity = 0.25

    elif cycle_position < 9:
        # Increasing traffic
        intensity = 0.25 + (
            (cycle_position - 4) / 4
        ) * 0.45

    elif cycle_position < 13:
        # High traffic / surge
        intensity = 0.85

    elif cycle_position < 17:
        # Traffic gradually decreases
        intensity = 0.85 - (
            (cycle_position - 12) / 4
        ) * 0.60

    else:
        # Low traffic
        intensity = 0.20

    # Add small random variation
    intensity += rng.normal(0, 0.035)
    intensity = float(np.clip(intensity, 0.08, 0.98))

    cpu_usage = np.clip(
        8 + intensity * 85 + rng.normal(0, 3),
        1,
        100
    )

    memory_usage = np.clip(
        15 + intensity * 75 + rng.normal(0, 3),
        1,
        100
    )

    active_users = max(
        10,
        int(50 + intensity * 2200 + rng.normal(0, 60))
    )

    requests_per_second = max(
        5,
        float(20 + intensity * 1400 + rng.normal(0, 35))
    )

    network_traffic_mb = max(
        1,
        float(10 + intensity * 600 + rng.normal(0, 20))
    )

    return {
        "cpu_usage": round(float(cpu_usage), 2),
        "memory_usage": round(float(memory_usage), 2),
        "active_users": active_users,
        "requests_per_second": round(
            float(requests_per_second), 2
        ),
        "network_traffic_mb": round(
            float(network_traffic_mb), 2
        )
    }


# ---------------------------------------------------------
# SCALING DECISION ENGINE
# ---------------------------------------------------------

def decide_scaling(
    predicted_cpu,
    current_instances,
    scale_out_threshold=60,
    scale_in_threshold=30,
    min_instances=1,
    max_instances=8
):
    """
    Decides whether to scale out, scale in, or maintain
    resources based on predicted CPU utilization.
    """

    if (
        predicted_cpu >= scale_out_threshold
        and current_instances < max_instances
    ):
        action = "Scale Out"
        recommended_instances = current_instances + 1

        reason = (
            f"Predicted CPU is {predicted_cpu:.2f}%, "
            f"which is at or above the "
            f"{scale_out_threshold}% scale-out threshold."
        )

    elif (
        predicted_cpu <= scale_in_threshold
        and current_instances > min_instances
    ):
        action = "Scale In"
        recommended_instances = current_instances - 1

        reason = (
            f"Predicted CPU is {predicted_cpu:.2f}%, "
            f"which is at or below the "
            f"{scale_in_threshold}% scale-in threshold."
        )

    else:
        action = "Maintain"
        recommended_instances = current_instances

        reason = (
            "Predicted CPU is within the configured scaling "
            "range, or the instance limit has been reached."
        )

    return {
        "action": action,
        "current_instances": current_instances,
        "recommended_instances": recommended_instances,
        "reason": reason
    }


# ---------------------------------------------------------
# SCALING EVENT HISTORY
# ---------------------------------------------------------

def create_event_record(
    step,
    workload,
    predicted_cpu,
    decision,
    timestamp=None
):
    """
    Creates a structured record for the scaling event
    timeline.
    """

    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return {
        "Step": step,
        "Timestamp": timestamp,
        "CPU Usage (%)": workload["cpu_usage"],
        "Predicted CPU (%)": round(predicted_cpu, 2),
        "Memory Usage (%)": workload["memory_usage"],
        "Active Users": workload["active_users"],
        "Requests/sec": workload["requests_per_second"],
        "Action": decision["action"],
        "Previous Instances": decision["current_instances"],
        "Current Instances": decision["recommended_instances"],
        "Reason": decision["reason"]
    }


# ---------------------------------------------------------
# CONVERT EVENT HISTORY TO DATAFRAME
# ---------------------------------------------------------

def get_event_dataframe(events):
    """
    Converts the stored event history into a DataFrame
    for dashboard tables and charts.
    """

    if not events:
        return pd.DataFrame()

    return pd.DataFrame(events)
