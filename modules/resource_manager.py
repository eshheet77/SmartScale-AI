import pandas as pd

from modules.ml_predictor import (
    prepare_features,
    predict_cpu
)


class AutoScaler:

    def __init__(
        self,
        initial_instances=2,
        min_instances=1,
        max_instances=8,
        scale_out_threshold=60,
        scale_in_threshold=30,
        cooldown=5
    ):

        self.instances = initial_instances

        self.min_instances = min_instances
        self.max_instances = max_instances

        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold

        self.cooldown = cooldown
        self.last_scaled_step = -cooldown

    def evaluate(self, cpu_usage, step):

        action = "No Change"

        # Enforce cooldown
        if step - self.last_scaled_step < self.cooldown:
            return self.instances, action

        # Scale out when CPU utilization is high
        if (
            cpu_usage >= self.scale_out_threshold
            and self.instances < self.max_instances
        ):

            self.instances += 1
            action = "Scale Out"

            self.last_scaled_step = step

        # Scale in when CPU utilization is low
        elif (
            cpu_usage <= self.scale_in_threshold
            and self.instances > self.min_instances
        ):

            self.instances -= 1
            action = "Scale In"

            self.last_scaled_step = step

        return self.instances, action


def simulate_scaling(
    model,
    dataset_path="data/cloud_workload.csv"
):
    """
    Simulate and compare predictive and reactive scaling.

    Predictive scaling uses the ML forecast.
    Reactive scaling uses current CPU utilization.

    Both strategies use identical scaling thresholds.
    """

    data = pd.read_csv(dataset_path)

    data = prepare_features(data)

    # Use the final 20% of chronological records
    split_index = int(len(data) * 0.8)

    test_data = data.iloc[
        split_index:
    ].reset_index(drop=True)

    # Independent autoscalers
    predictive_scaler = AutoScaler()
    reactive_scaler = AutoScaler()

    results = []

    for step, row in test_data.iterrows():

        current_cpu = float(row["cpu_usage"])

        workload = row.to_dict()

        # ML prediction with fallback
        prediction_valid = True

        try:

            predicted_cpu = predict_cpu(
                model,
                workload
            )

            if not (
                isinstance(predicted_cpu, (int, float))
                and 0 <= predicted_cpu <= 100
            ):

                raise ValueError(
                    "Invalid prediction"
                )

        except Exception:

            prediction_valid = False

            # Fallback to reactive CPU reading
            predicted_cpu = current_cpu

        # Predictive scaling
        predictive_instances, predictive_action = (
            predictive_scaler.evaluate(
                predicted_cpu,
                step
            )
        )

        # Reactive scaling
        reactive_instances, reactive_action = (
            reactive_scaler.evaluate(
                current_cpu,
                step
            )
        )

        results.append({

            "timestamp": row["timestamp"],

            "current_cpu": current_cpu,

            "predicted_cpu": predicted_cpu,

            "actual_future_cpu": float(
                row["future_cpu_usage"]
            ),

            "prediction_valid": prediction_valid,

            "predictive_instances": predictive_instances,

            "reactive_instances": reactive_instances,

            "predictive_action": predictive_action,

            "reactive_action": reactive_action

        })

    return pd.DataFrame(results)
