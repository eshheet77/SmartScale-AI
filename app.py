
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.ml_predictor import load_model, train_model, predict_cpu
from modules.resource_manager import simulate_scaling
from modules.monitor import (
    add_monitoring_metrics,
    calculate_monitoring_summary
)
from modules.evaluator import generate_evaluation_report

from modules.live_simulator import (
    generate_workload,
    decide_scaling,
    create_event_record,
    get_event_dataframe
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartScale AI",
    page_icon="☁️",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    [data-testid="stMetric"] {
        background-color: #1c2333;
        border: 1px solid #30394d;
        padding: 15px;
        border-radius: 10px;
    }

    .section-note {
        color: #aab4c5;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL AND DATA LOADING
# =========================================================

@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_dataset():
    return pd.read_csv("data/cloud_workload.csv")


@st.cache_data
def get_results():
    model = get_model()

    results = simulate_scaling(
        model,
        dataset_path="data/cloud_workload.csv"
    )

    results = add_monitoring_metrics(results)

    report = generate_evaluation_report(results)

    return results, report


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_inr(value):
    return f"₹{value:,.2f}"


def make_line_chart(data, x, y, title, labels=None):

    fig = px.line(
        data,
        x=x,
        y=y,
        title=title,
        labels=labels
    )

    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20)
    )

    return fig


def get_live_prediction(model, workload):

    prediction = predict_cpu(model, workload)

    if isinstance(prediction, (list, tuple, np.ndarray)):
        prediction = np.asarray(prediction).flatten()[0]

    prediction = float(prediction)

    if not np.isfinite(prediction):
        raise ValueError("Invalid prediction returned by the model.")

    return float(np.clip(prediction, 0, 100))


def initialize_simulation_state():

    defaults = {
        "live_step": 0,
        "live_current_instances": 2,
        "live_current_workload": None,
        "live_workload_history": [],
        "live_event_history": [],
        "live_prediction_history": [],
        "live_processed": False,
        "live_latest_result": None
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


def reset_simulation():

    st.session_state.live_step = 0
    st.session_state.live_current_instances = 2
    st.session_state.live_current_workload = None
    st.session_state.live_workload_history = []
    st.session_state.live_event_history = []
    st.session_state.live_prediction_history = []
    st.session_state.live_processed = False
    st.session_state.live_latest_result = None


def run_scaling_simulation(model, workload, source):

    predicted_cpu = get_live_prediction(
        model,
        workload
    )

    current_instances = st.session_state.live_current_instances

    decision = decide_scaling(
        predicted_cpu,
        current_instances
    )

    # Record the event before updating the instance count.
    event = create_event_record(
        step=st.session_state.live_step,
        workload=workload,
        predicted_cpu=predicted_cpu,
        decision=decision
    )

    event["Source"] = source

    st.session_state.live_event_history.append(event)

    st.session_state.live_current_instances = (
        decision["recommended_instances"]
    )

    prediction_record = {
        "Step": st.session_state.live_step,
        "Timestamp": event["Timestamp"],
        "Current CPU": workload["cpu_usage"],
        "Predicted CPU": predicted_cpu,
        "Current Instances": current_instances,
        "Recommended Instances": decision["recommended_instances"],
        "Action": decision["action"],
        "Source": source
    }

    st.session_state.live_prediction_history.append(
        prediction_record
    )

    st.session_state.live_latest_result = {
        "workload": workload,
        "predicted_cpu": predicted_cpu,
        "decision": decision,
        "source": source
    }

    st.session_state.live_processed = True


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("☁️ SmartScale AI")

st.sidebar.markdown(
    "### ML-Based Cloud Resource Prediction"
)

st.sidebar.caption(
    "Intelligent workload prediction and auto-scaling simulation"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Workload Monitoring",
        "ML Predictions",
        "Auto Scaling",
        "Performance Evaluation",
        "Model Information",
        "Live Simulation"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("**System Configuration**")
st.sidebar.write("ML Model: Random Forest")
st.sidebar.write("Scaling Mode: Predictive + Reactive")
st.sidebar.write("Environment: Local Simulation")

st.sidebar.caption("SmartScale AI | Academic Project")


# =========================================================
# LOAD COMMON DATA
# =========================================================

try:

    model = get_model()
    dataset = get_dataset()
    results, report = get_results()

except Exception as error:

    st.error(f"Unable to load project data: {error}")

    st.info(
        "Check that the dataset and trained model exist. "
        "Generate the dataset and train the model if necessary."
    )

    st.stop()


# =========================================================
# 1. OVERVIEW
# =========================================================

if page == "Overview":

    st.title("☁️ SmartScale AI Dashboard")

    st.markdown(
        "### ML-Based Cloud Resource Prediction and Intelligent Auto-Scaling"
    )

    st.markdown(
        "SmartScale AI predicts upcoming cloud workload demand "
        "and simulates intelligent resource allocation."
    )

    total_records = len(dataset)

    avg_cpu = dataset["cpu_usage"].mean()
    max_cpu = dataset["cpu_usage"].max()

    prediction_metrics = report["prediction_metrics"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Workload Records", f"{total_records:,}")
    col2.metric("Average CPU", f"{avg_cpu:.2f}%")
    col3.metric("Peak CPU", f"{max_cpu:.2f}%")

    col4.metric(
        "Prediction MAE",
        f"{prediction_metrics['MAE']:.2f}%"
    )

    st.divider()

    st.subheader("System Architecture")

    st.markdown("""
    **Workload Generator**
    ↓

    **Monitoring and Data Collection**
    ↓

    **Random Forest ML Prediction**
    ↓

    **Predictive Auto-Scaling Engine**
    ↓

    **Performance and Cost Evaluation**
    """)

    st.subheader("CPU Workload Overview")

    st.plotly_chart(
        make_line_chart(
            dataset,
            "timestamp",
            "cpu_usage",
            "Historical CPU Utilization"
        ),
        use_container_width=True
    )

    st.subheader("Project Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🤖 ML Prediction")
        st.write("Forecasts future CPU utilization.")

    with col2:
        st.markdown("#### ⚙️ Auto-Scaling")
        st.write("Simulates resource scaling decisions.")

    with col3:
        st.markdown("#### 📊 Evaluation")
        st.write("Compares predictive and reactive scaling.")


# =========================================================
# 2. WORKLOAD MONITORING
# =========================================================

elif page == "Workload Monitoring":

    st.title("📊 Workload Monitoring")

    st.write(
        "Monitor the simulated cloud environment's workload metrics."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "CPU Usage",
        f"{dataset['cpu_usage'].iloc[-1]:.2f}%"
    )

    col2.metric(
        "Memory Usage",
        f"{dataset['memory_usage'].iloc[-1]:.2f}%"
    )

    col3.metric(
        "Active Users",
        f"{int(dataset['active_users'].iloc[-1]):,}"
    )

    col4.metric(
        "Requests / Second",
        f"{dataset['requests_per_second'].iloc[-1]:.0f}"
    )

    st.divider()

    st.subheader("CPU Utilization")

    st.plotly_chart(
        make_line_chart(
            dataset,
            "timestamp",
            "cpu_usage",
            "CPU Usage Over Time"
        ),
        use_container_width=True
    )

    st.subheader("Memory Utilization")

    st.plotly_chart(
        make_line_chart(
            dataset,
            "timestamp",
            "memory_usage",
            "Memory Usage Over Time"
        ),
        use_container_width=True
    )

    st.subheader("Active Users")

    st.plotly_chart(
        make_line_chart(
            dataset,
            "timestamp",
            "active_users",
            "Active Users Over Time"
        ),
        use_container_width=True
    )

    st.subheader("Workload Data")

    st.dataframe(
        dataset.tail(100),
        use_container_width=True
    )


# =========================================================
# 3. ML PREDICTIONS
# =========================================================

elif page == "ML Predictions":

    st.title("🤖 Machine Learning Predictions")

    st.write(
        "The Random Forest model predicts CPU utilization "
        "five minutes into the future."
    )

    prediction_metrics = report["prediction_metrics"]

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{prediction_metrics['MAE']:.4f}")
    col2.metric("RMSE", f"{prediction_metrics['RMSE']:.4f}")

    # FIXED: Actual dictionary key is R2 Score.
    col3.metric(
        "R² Score",
        f"{prediction_metrics['R2 Score']:.4f}"
    )

    st.divider()

    prediction_data = results.tail(200).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prediction_data["timestamp"],
            y=prediction_data["actual_future_cpu"],
            name="Actual Future CPU",
            mode="lines"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=prediction_data["timestamp"],
            y=prediction_data["predicted_cpu"],
            name="Predicted CPU",
            mode="lines"
        )
    )

    fig.update_layout(
        title="Actual vs Predicted CPU Utilization",
        template="plotly_dark",
        height=430,
        xaxis_title="Timestamp",
        yaxis_title="CPU Utilization (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Records")

    st.dataframe(
        prediction_data[
            [
                "timestamp",
                "current_cpu",
                "predicted_cpu",
                "actual_future_cpu"
            ]
        ],
        use_container_width=True
    )


# =========================================================
# 4. AUTO SCALING
# =========================================================

elif page == "Auto Scaling":

    st.title("⚙️ Intelligent Auto-Scaling")

    st.write(
        "Compare ML-based predictive scaling with "
        "traditional threshold-based reactive scaling."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Predictive Scaling")
        st.write("Uses predicted future CPU demand.")

        st.metric(
            "Average Instances",
            f"{results['predictive_instances'].mean():.2f}"
        )

        st.metric(
            "Scale Out Actions",
            int(
                (results["predictive_action"] == "Scale Out").sum()
            )
        )

    with col2:

        st.subheader("Reactive Scaling")
        st.write("Uses current CPU utilization.")

        st.metric(
            "Average Instances",
            f"{results['reactive_instances'].mean():.2f}"
        )

        st.metric(
            "Scale Out Actions",
            int(
                (results["reactive_action"] == "Scale Out").sum()
            )
        )

    st.divider()

    scaling_data = results.tail(250)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scaling_data["timestamp"],
            y=scaling_data["predictive_instances"],
            name="Predictive Instances"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=scaling_data["timestamp"],
            y=scaling_data["reactive_instances"],
            name="Reactive Instances"
        )
    )

    fig.update_layout(
        title="Predictive vs Reactive Instance Allocation",
        template="plotly_dark",
        height=420,
        xaxis_title="Timestamp",
        yaxis_title="Number of Instances"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scaling Event History")

    event_data = results[
        (results["predictive_action"] != "No Change")
        | (results["reactive_action"] != "No Change")
    ]

    st.dataframe(
        event_data[
            [
                "timestamp",
                "current_cpu",
                "predicted_cpu",
                "predictive_action",
                "predictive_instances",
                "reactive_action",
                "reactive_instances"
            ]
        ].tail(100),
        use_container_width=True
    )


# =========================================================
# 5. PERFORMANCE EVALUATION
# =========================================================

elif page == "Performance Evaluation":

    st.title("📈 Performance Evaluation")

    st.write(
        "Compare the estimated cost, energy, latency and "
        "resource utilization of both scaling strategies."
    )

    comparison = report["strategy_comparison"]

    st.subheader("Strategy Comparison")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.divider()

    savings = report["savings"]

    col1, col2, col3, col4 = st.columns(4)

    # FIXED: Use the exact dictionary keys returned by evaluator.py.
    col1.metric(
        "Estimated Cost Savings",
        f"{savings['Cost Savings (%)']:.2f}%"
    )

    col2.metric(
        "Estimated Energy Savings",
        f"{savings['Energy Savings (%)']:.2f}%"
    )

    col3.metric(
        "Latency Improvement",
        f"{savings['Average Latency Improvement (%)']:.2f}%"
    )

    col4.metric(
        "Overload Reduction",
        f"{savings['Overload Reduction (%)']:.2f}%"
    )

    st.caption(
        "Cost and energy figures are estimates from the local "
        "simulation, not actual cloud billing or physical measurements."
    )

    st.caption(
        "Negative latency improvement or overload reduction means "
        "the predictive strategy performed worse than reactive scaling "
        "for that particular metric in this simulation."
    )


# =========================================================
# 6. MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.title("🧠 Model Information")

    st.markdown("### Random Forest Regression")

    st.write(
        "SmartScale AI uses a Random Forest Regressor to predict "
        "future CPU utilization from historical workload information."
    )

    prediction_metrics = report["prediction_metrics"]

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{prediction_metrics['MAE']:.4f}")
    col2.metric("RMSE", f"{prediction_metrics['RMSE']:.4f}")

    # FIXED: Actual dictionary key is R2 Score.
    col3.metric(
        "R² Score",
        f"{prediction_metrics['R2 Score']:.4f}"
    )

    st.divider()

    st.subheader("Model Feature Importance")

    try:

        importance = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=True
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as error:

        st.warning(
            f"Feature importance unavailable: {error}"
        )

    st.divider()

    if st.button("Retrain Machine Learning Model"):

        with st.spinner("Training Random Forest model..."):

            train_model()

        st.success("Model training completed.")

        get_model.clear()
        get_results.clear()

        st.rerun()


# =========================================================
# 7. LIVE SIMULATION
# =========================================================

elif page == "Live Simulation":

    initialize_simulation_state()

    st.title("🚀 SmartScale AI - Live Simulation")

    st.markdown(
        "Generate cloud workloads, predict upcoming CPU demand, "
        "and observe intelligent resource scaling."
    )

    st.info(
        "This is a local cloud simulation. The trained Random Forest "
        "model makes predictions, while the scaling engine recommends "
        "instance changes. No real cloud resources are deployed."
    )

    tab1, tab2 = st.tabs([
        "⚡ Live Workload Generator",
        "✍️ Manual Workload Testing"
    ])

    # -----------------------------------------------------
    # TAB 1: AUTOMATIC WORKLOAD GENERATOR
    # -----------------------------------------------------

    with tab1:

        st.subheader("1. Generate Simulated Cloud Traffic")

        st.write(
            "Generate changing workloads that follow normal traffic, "
            "increasing traffic, traffic surges and cooldown periods."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Simulation Step",
            st.session_state.live_step
        )

        col2.metric(
            "Running Instances",
            st.session_state.live_current_instances
        )

        col3.metric(
            "Generated Workloads",
            len(st.session_state.live_workload_history)
        )

        button_col1, button_col2 = st.columns(2)

        with button_col1:

            if st.button(
                "⚡ Generate Next Workload",
                use_container_width=True
            ):

                workload = generate_workload(
                    st.session_state.live_step
                )

                st.session_state.live_current_workload = workload

                st.session_state.live_workload_history.append({
                    "Step": st.session_state.live_step,
                    "Timestamp": pd.Timestamp.now(),
                    **workload
                })

                st.session_state.live_step += 1

                st.session_state.live_processed = False

                st.session_state.live_latest_result = None

        with button_col2:

            if st.button(
                "🔄 Reset Simulation",
                use_container_width=True
            ):

                reset_simulation()
                st.rerun()

        current_workload = st.session_state.live_current_workload

        if current_workload is not None:

            st.divider()

            st.subheader("2. Current Generated Workload")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "CPU Usage",
                f"{current_workload['cpu_usage']:.2f}%"
            )

            col2.metric(
                "Memory Usage",
                f"{current_workload['memory_usage']:.2f}%"
            )

            col3.metric(
                "Active Users",
                f"{current_workload['active_users']:,}"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Requests / Second",
                f"{current_workload['requests_per_second']:.2f}"
            )

            col2.metric(
                "Network Traffic",
                f"{current_workload['network_traffic_mb']:.2f} MB"
            )

            if not st.session_state.live_processed:

                if st.button(
                    "🤖 Run ML Prediction & Auto-Scaling",
                    type="primary",
                    use_container_width=True
                ):

                    try:

                        with st.spinner(
                            "Predicting future workload and evaluating scaling..."
                        ):

                            run_scaling_simulation(
                                model,
                                current_workload,
                                "Generated Workload"
                            )

                        st.rerun()

                    except Exception as error:

                        st.error(
                            f"Prediction or scaling failed: {error}"
                        )

            else:

                st.success(
                    "Prediction and scaling decision completed for this workload."
                )

            # Latest result

            latest = st.session_state.live_latest_result

            if latest is not None:

                decision = latest["decision"]

                st.divider()

                st.subheader("3. ML Prediction Result")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Current CPU",
                    f"{current_workload['cpu_usage']:.2f}%"
                )

                col2.metric(
                    "Predicted Future CPU",
                    f"{latest['predicted_cpu']:.2f}%"
                )

                col3.metric(
                    "Current Instances",
                    decision["current_instances"]
                )

                st.subheader("4. Auto-Scaling Decision")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Recommended Instances",
                    decision["recommended_instances"]
                )

                col2.metric(
                    "Scaling Action",
                    decision["action"]
                )

                if decision["action"] == "Scale Out":

                    st.success(
                        "SCALE OUT: Additional resources are recommended."
                    )

                elif decision["action"] == "Scale In":

                    st.warning(
                        "SCALE IN: Resource allocation can be reduced."
                    )

                else:

                    st.info(
                        "MAINTAIN: Current resource allocation is sufficient."
                    )

                st.write("**Decision Reason:**")
                st.write(decision["reason"])

        else:

            st.info(
                "Click Generate Next Workload to start the simulation."
            )

        # -------------------------------------------------
        # WORKLOAD HISTORY AND CHARTS
        # -------------------------------------------------

        if st.session_state.live_workload_history:

            st.divider()

            st.subheader("5. Live Workload History")

            workload_history = pd.DataFrame(
                st.session_state.live_workload_history
            )

            metric_options = {
                "CPU Usage": "cpu_usage",
                "Memory Usage": "memory_usage",
                "Active Users": "active_users",
                "Requests Per Second": "requests_per_second",
                "Network Traffic": "network_traffic_mb"
            }

            selected_metric = st.selectbox(
                "Select workload metric",
                list(metric_options.keys()),
                key="live_metric_select"
            )

            fig = px.line(
                workload_history,
                x="Step",
                y=metric_options[selected_metric],
                markers=True,
                title=f"{selected_metric} During Simulation"
            )

            fig.update_layout(
                template="plotly_dark",
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                workload_history.tail(50),
                use_container_width=True
            )

        # -------------------------------------------------
        # PREDICTION AND INSTANCE HISTORY
        # -------------------------------------------------

        if st.session_state.live_prediction_history:

            st.divider()

            st.subheader("6. Prediction and Instance History")

            prediction_history = pd.DataFrame(
                st.session_state.live_prediction_history
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=prediction_history["Step"],
                    y=prediction_history["Predicted CPU"],
                    mode="lines+markers",
                    name="Predicted CPU"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=prediction_history["Step"],
                    y=prediction_history["Current CPU"],
                    mode="lines+markers",
                    name="Current CPU"
                )
            )

            fig.update_layout(
                title="Current vs Predicted CPU",
                template="plotly_dark",
                height=350,
                xaxis_title="Simulation Step",
                yaxis_title="CPU (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            instance_fig = go.Figure()

            instance_fig.add_trace(
                go.Scatter(
                    x=prediction_history["Step"],
                    y=prediction_history["Recommended Instances"],
                    mode="lines+markers",
                    name="Running Instances"
                )
            )

            instance_fig.update_layout(
                title="Instance Count During Simulation",
                template="plotly_dark",
                height=350,
                xaxis_title="Simulation Step",
                yaxis_title="Number of Instances"
            )

            st.plotly_chart(
                instance_fig,
                use_container_width=True
            )

    # -----------------------------------------------------
    # TAB 2: MANUAL WORKLOAD TESTING
    # -----------------------------------------------------

    with tab2:

        st.subheader("Manual Workload Testing")

        st.write(
            "Enter custom cloud workload values and test the trained model."
        )

        scenarios = {
            "Custom Workload": {
                "cpu": 45.0,
                "memory": 60.0,
                "users": 500,
                "requests": 250.0,
                "network": 100.0
            },
            "Normal Traffic": {
                "cpu": 25.0,
                "memory": 40.0,
                "users": 100,
                "requests": 50.0,
                "network": 25.0
            },
            "Moderate Traffic": {
                "cpu": 50.0,
                "memory": 60.0,
                "users": 500,
                "requests": 250.0,
                "network": 100.0
            },
            "High Traffic": {
                "cpu": 75.0,
                "memory": 80.0,
                "users": 1000,
                "requests": 600.0,
                "network": 250.0
            },
            "Traffic Surge": {
                "cpu": 90.0,
                "memory": 95.0,
                "users": 2000,
                "requests": 1200.0,
                "network": 500.0
            },
            "Low Traffic": {
                "cpu": 15.0,
                "memory": 25.0,
                "users": 50,
                "requests": 20.0,
                "network": 10.0
            }
        }

        selected_scenario = st.selectbox(
            "Choose a workload scenario",
            list(scenarios.keys()),
            key="manual_scenario"
        )

        selected_values = scenarios[selected_scenario]

        with st.form("manual_workload_form"):

            col1, col2 = st.columns(2)

            with col1:

                cpu_input = st.slider(
                    "Current CPU Usage (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(selected_values["cpu"]),
                    step=1.0
                )

                memory_input = st.slider(
                    "Memory Usage (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(selected_values["memory"]),
                    step=1.0
                )

                users_input = st.number_input(
                    "Active Users",
                    min_value=0,
                    max_value=100000,
                    value=int(selected_values["users"]),
                    step=50
                )

            with col2:

                requests_input = st.number_input(
                    "Requests Per Second",
                    min_value=0.0,
                    max_value=100000.0,
                    value=float(selected_values["requests"]),
                    step=10.0
                )

                network_input = st.number_input(
                    "Network Traffic (MB)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=float(selected_values["network"]),
                    step=10.0
                )

                current_instances_input = st.number_input(
                    "Current Running Instances",
                    min_value=1,
                    max_value=8,
                    value=2,
                    step=1
                )

            manual_submit = st.form_submit_button(
                "▶ Run Manual Simulation",
                use_container_width=True
            )

        if manual_submit:

            workload = {
                "cpu_usage": float(cpu_input),
                "memory_usage": float(memory_input),
                "active_users": int(users_input),
                "requests_per_second": float(requests_input),
                "network_traffic_mb": float(network_input)
            }

            try:

                predicted_cpu = get_live_prediction(
                    model,
                    workload
                )

                decision = decide_scaling(
                    predicted_cpu,
                    int(current_instances_input)
                )

                st.session_state.manual_latest_result = {
                    "workload": workload,
                    "predicted_cpu": predicted_cpu,
                    "decision": decision
                }

            except Exception as error:

                st.error(
                    f"Manual simulation failed: {error}"
                )

        if "manual_latest_result" in st.session_state:

            manual = st.session_state.manual_latest_result

            st.divider()

            st.subheader("Manual Simulation Result")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Current CPU",
                f"{manual['workload']['cpu_usage']:.2f}%"
            )

            col2.metric(
                "Predicted CPU",
                f"{manual['predicted_cpu']:.2f}%"
            )

            col3.metric(
                "Recommended Instances",
                manual["decision"]["recommended_instances"]
            )

            st.metric(
                "Scaling Action",
                manual["decision"]["action"]
            )

            st.write(
                "**Reason:**",
                manual["decision"]["reason"]
            )


# =========================================================
# SCALING EVENT TIMELINE
# =========================================================

if page == "Live Simulation":

    st.divider()

    st.title("📜 Scaling Event Timeline")

    events = get_event_dataframe(
        st.session_state.live_event_history
    )

    if not events.empty:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Scaling Decisions",
            len(events)
        )

        col2.metric(
            "Scale Out Events",
            int((events["Action"] == "Scale Out").sum())
        )

        col3.metric(
            "Scale In Events",
            int((events["Action"] == "Scale In").sum())
        )

        st.subheader("Event History")

        st.dataframe(
            events.sort_values(
                "Step",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Scaling Actions")

        action_counts = (
            events["Action"]
            .value_counts()
            .rename_axis("Action")
            .reset_index(name="Count")
        )

        fig = px.bar(
            action_counts,
            x="Action",
            y="Count",
            color="Action",
            title="Scaling Decision Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No scaling events recorded yet. Generate a workload "
            "and run ML prediction to start building the timeline."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "SmartScale AI | ML-Based Cloud Resource Prediction and "
    "Intelligent Auto-Scaling System | Local Simulation"
)
