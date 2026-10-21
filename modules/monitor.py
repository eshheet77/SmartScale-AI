import numpy as np
import pandas as pd


def add_monitoring_metrics(
    results,
    instance_hourly_cost_inr=5.0,
    idle_power_kw=0.05,
    max_power_kw=0.20,
    baseline_instances=2
):
    """
    Calculate simulated monitoring metrics for both
    predictive and reactive autoscaling strategies.

    Assumptions:
    - Initial baseline: 2 instances.
    - Each instance costs Rs. 5 per simulated hour.
    - Idle power per instance: 0.05 kW.
    - Maximum power per instance: 0.20 kW.

    These are illustrative simulation assumptions,
    not actual cloud-provider prices or measurements.
    """

    data = results.copy()

    # Avoid division by zero
    predictive_instances = (
        data["predictive_instances"].clip(lower=1)
    )

    reactive_instances = (
        data["reactive_instances"].clip(lower=1)
    )

    # ------------------------------------------------
    # 1. ESTIMATED CPU UTILIZATION PER INSTANCE
    # ------------------------------------------------

    data["predictive_utilization"] = np.clip(
        data["current_cpu"]
        * baseline_instances
        / predictive_instances,
        0,
        100
    )

    data["reactive_utilization"] = np.clip(
        data["current_cpu"]
        * baseline_instances
        / reactive_instances,
        0,
        100
    )

    # ------------------------------------------------
    # 2. ESTIMATED RESPONSE LATENCY
    # ------------------------------------------------

    def estimate_latency(utilization):

        base_latency = 50

        normal_load_penalty = (
            np.maximum(utilization - 60, 0) * 2.5
        )

        high_load_penalty = (
            np.maximum(utilization - 85, 0) * 5
        )

        return (
            base_latency
            + normal_load_penalty
            + high_load_penalty
        )

    data["predictive_latency_ms"] = (
        estimate_latency(
            data["predictive_utilization"]
        )
    )

    data["reactive_latency_ms"] = (
        estimate_latency(
            data["reactive_utilization"]
        )
    )

    # ------------------------------------------------
    # 3. ESTIMATED OPERATING COST
    # ------------------------------------------------

    cost_per_minute = (
        instance_hourly_cost_inr / 60
    )

    data["predictive_cost_inr"] = (
        predictive_instances * cost_per_minute
    )

    data["reactive_cost_inr"] = (
        reactive_instances * cost_per_minute
    )

    # ------------------------------------------------
    # 4. ESTIMATED ENERGY CONSUMPTION
    # ------------------------------------------------

    def estimate_energy(instances, utilization):

        # Estimated power per instance
        power_per_instance = (
            idle_power_kw
            + (max_power_kw - idle_power_kw)
            * utilization / 100
        )

        # Total cluster power
        total_power_kw = (
            instances * power_per_instance
        )

        # One simulation record represents one minute
        energy_kwh = total_power_kw / 60

        return energy_kwh

    data["predictive_energy_kwh"] = (
        estimate_energy(
            predictive_instances,
            data["predictive_utilization"]
        )
    )

    data["reactive_energy_kwh"] = (
        estimate_energy(
            reactive_instances,
            data["reactive_utilization"]
        )
    )

    # ------------------------------------------------
    # 5. OVERLOAD DETECTION
    # ------------------------------------------------

    data["predictive_overload"] = (
        data["predictive_utilization"] >= 80
    )

    data["reactive_overload"] = (
        data["reactive_utilization"] >= 80
    )

    return data


def calculate_monitoring_summary(data):

    """
    Generate summary metrics for both scaling strategies.
    """

    summary = {

        "Predictive Scaling": {

            "Average CPU Utilization (%)":
                data["predictive_utilization"].mean(),

            "Maximum CPU Utilization (%)":
                data["predictive_utilization"].max(),

            "Average Latency (ms)":
                data["predictive_latency_ms"].mean(),

            "Maximum Latency (ms)":
                data["predictive_latency_ms"].max(),

            "Total Cost (INR)":
                data["predictive_cost_inr"].sum(),

            "Total Energy (kWh)":
                data["predictive_energy_kwh"].sum(),

            "Overload Minutes":
                int(data["predictive_overload"].sum()),

            "Average Instances":
                data["predictive_instances"].mean()

        },

        "Reactive Scaling": {

            "Average CPU Utilization (%)":
                data["reactive_utilization"].mean(),

            "Maximum CPU Utilization (%)":
                data["reactive_utilization"].max(),

            "Average Latency (ms)":
                data["reactive_latency_ms"].mean(),

            "Maximum Latency (ms)":
                data["reactive_latency_ms"].max(),

            "Total Cost (INR)":
                data["reactive_cost_inr"].sum(),

            "Total Energy (kWh)":
                data["reactive_energy_kwh"].sum(),

            "Overload Minutes":
                int(data["reactive_overload"].sum()),

            "Average Instances":
                data["reactive_instances"].mean()

        }

    }

    return pd.DataFrame(summary).round(3)