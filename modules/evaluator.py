import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_prediction_metrics(data):
    """
    Evaluate ML prediction accuracy using the actual
    future CPU utilization as the reference.
    """

    actual = data["actual_future_cpu"]
    predicted = data["predicted_cpu"]

    metrics = {
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),

        "R2 Score": r2_score(
            actual,
            predicted
        )
    }

    return {
        key: round(float(value), 4)
        for key, value in metrics.items()
    }


def evaluate_strategies(data):
    """
    Compare predictive and reactive autoscaling
    using monitoring results.
    """

    strategies = {
        "Predictive Scaling": {
            "instances": "predictive_instances",
            "utilization": "predictive_utilization",
            "latency": "predictive_latency_ms",
            "cost": "predictive_cost_inr",
            "energy": "predictive_energy_kwh",
            "overload": "predictive_overload",
            "action": "predictive_action"
        },

        "Reactive Scaling": {
            "instances": "reactive_instances",
            "utilization": "reactive_utilization",
            "latency": "reactive_latency_ms",
            "cost": "reactive_cost_inr",
            "energy": "reactive_energy_kwh",
            "overload": "reactive_overload",
            "action": "reactive_action"
        }
    }

    evaluation = {}

    for name, columns in strategies.items():

        actions = data[columns["action"]]

        evaluation[name] = {

            "Average Instances":
                data[columns["instances"]].mean(),

            "Maximum Instances":
                data[columns["instances"]].max(),

            "Average CPU Utilization (%)":
                data[columns["utilization"]].mean(),

            "Maximum CPU Utilization (%)":
                data[columns["utilization"]].max(),

            "Average Latency (ms)":
                data[columns["latency"]].mean(),

            "Maximum Latency (ms)":
                data[columns["latency"]].max(),

            "Estimated Cost (INR)":
                data[columns["cost"]].sum(),

            "Estimated Energy (kWh)":
                data[columns["energy"]].sum(),

            "Overload Minutes":
                int(data[columns["overload"]].sum()),

            "Scale Out Actions":
                int((actions == "Scale Out").sum()),

            "Scale In Actions":
                int((actions == "Scale In").sum())

        }

    return pd.DataFrame(evaluation).round(3)


def calculate_savings(data):
    """
    Calculate estimated savings of predictive scaling
    compared with reactive scaling.

    Positive percentage = savings.
    Negative percentage = additional consumption.
    """

    comparisons = {

        "Cost Savings (%)": (
            "predictive_cost_inr",
            "reactive_cost_inr"
        ),

        "Energy Savings (%)": (
            "predictive_energy_kwh",
            "reactive_energy_kwh"
        ),

        "Average Latency Improvement (%)": (
            "predictive_latency_ms",
            "reactive_latency_ms"
        ),

        "Overload Reduction (%)": (
            "predictive_overload",
            "reactive_overload"
        )

    }

    savings = {}

    for name, (predictive_column, reactive_column) in comparisons.items():

        predictive_value = data[predictive_column].sum()
        reactive_value = data[reactive_column].sum()

        if reactive_value == 0:

            percentage = 0.0

        else:

            percentage = (
                (reactive_value - predictive_value)
                / reactive_value
            ) * 100

        savings[name] = round(
            float(percentage),
            2
        )

    return savings


def generate_evaluation_report(data):
    """
    Generate a complete evaluation report containing
    prediction accuracy, strategy comparison and savings.
    """

    prediction_metrics = (
        calculate_prediction_metrics(data)
    )

    strategy_comparison = (
        evaluate_strategies(data)
    )

    savings = calculate_savings(data)

    return {
        "prediction_metrics": prediction_metrics,
        "strategy_comparison": strategy_comparison,
        "savings": savings
    }
