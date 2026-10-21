import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Features used by the machine learning model
FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "active_users",
    "requests_per_second",
    "network_traffic_mb",
    "cpu_lag_1",
    "cpu_lag_5",
    "cpu_lag_10",
    "cpu_rolling_mean_5",
    "cpu_rolling_mean_10",
    "users_lag_1",
    "hour_sin",
    "hour_cos"
]


def prepare_features(data):
    """
    Create historical and time-based features.
    Only current and past information is used.
    """

    data = data.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    # Previous CPU readings
    data["cpu_lag_1"] = (
        data["cpu_usage"].shift(1)
    )

    data["cpu_lag_5"] = (
        data["cpu_usage"].shift(5)
    )

    data["cpu_lag_10"] = (
        data["cpu_usage"].shift(10)
    )

    # Historical CPU averages
    data["cpu_rolling_mean_5"] = (
        data["cpu_usage"].rolling(window=5).mean()
    )

    data["cpu_rolling_mean_10"] = (
        data["cpu_usage"].rolling(window=10).mean()
    )

    # Previous active-user count
    data["users_lag_1"] = (
        data["active_users"].shift(1)
    )

    # Time-based features
    hour = (
        data["timestamp"].dt.hour
        + data["timestamp"].dt.minute / 60
    )

    data["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    # Remove rows without enough historical information
    data = data.dropna().reset_index(drop=True)

    return data


def train_model(
    dataset_path="data/cloud_workload.csv"
):
    """
    Train a Random Forest model to predict CPU
    utilization five minutes into the future.
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    data = pd.read_csv(dataset_path)

    data = prepare_features(data)

    X = data[FEATURE_COLUMNS]
    y = data["future_cpu_usage"]

    # Chronological 80/20 split
    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Random Forest Regression model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict future CPU utilization
    predictions = model.predict(X_test)

    # Evaluate model performance
    metrics = {
        "MAE": float(
            mean_absolute_error(y_test, predictions)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_test, predictions)
            )
        ),

        "R2 Score": float(
            r2_score(y_test, predictions)
        )
    }

    # Save trained model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/random_forest_model.pkl"
    )

    # Store actual and predicted values
    results = pd.DataFrame({
        "Actual CPU": y_test.values,
        "Predicted CPU": predictions
    })

    # Feature importance
    feature_importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    return (
        model,
        metrics,
        results,
        feature_importance
    )


def load_model(
    model_path="models/random_forest_model.pkl"
):
    """
    Load a previously trained Random Forest model.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found. Train the model first."
        )

    return joblib.load(model_path)


def predict_cpu(model, workload):
    """
    Predict CPU utilization five minutes ahead.
    """

    input_data = pd.DataFrame(
        [workload],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(input_data)[0]

    return float(
        np.clip(prediction, 0, 100)
    )
