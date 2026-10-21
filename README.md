# SmartScale AI
### ML-Based Cloud Resource Prediction and Intelligent Auto-Scaling System

SmartScale AI is a machine learning-based cloud resource management project that predicts future CPU utilization and recommends intelligent resource scaling decisions based on workload patterns.

The system uses a Random Forest Regression model to predict CPU usage and compares predictive auto-scaling with traditional reactive auto-scaling. It provides an interactive dashboard for workload monitoring, predictions, scaling decisions, and performance evaluation.

**This project runs locally using simulated cloud workloads. It does not require AWS, Azure, paid cloud services, or cloud credentials.**

---

## 1. Problem Statement

Cloud applications experience changing workloads throughout the day. During high traffic, insufficient resources can cause performance issues, while maintaining unnecessary resources during low traffic increases operational costs.

Traditional reactive scaling responds only after resource utilization changes.

SmartScale AI addresses this problem by using machine learning to predict future CPU utilization and recommend scaling decisions in advance.

## 2. Objectives

- Predict future CPU utilization using machine learning.
- Monitor workload parameters such as CPU, memory, users, and requests.
- Implement intelligent predictive auto-scaling.
- Compare predictive scaling with reactive scaling.
- Estimate resource utilization, operational costs, and energy consumption.
- Provide an interactive dashboard for monitoring and evaluation.
- Demonstrate workload changes through live and manual simulations.

## 3. Key Features

### Workload Monitoring
- CPU utilization monitoring.
- Memory utilization monitoring.
- Active user tracking.
- Requests-per-second monitoring.
- Network traffic monitoring.
- Historical workload visualization.

### Machine Learning Predictions
- Random Forest Regression model.
- Future CPU utilization prediction.
- Actual vs predicted CPU comparison.
- Model performance evaluation.
- Feature importance visualization.

### Intelligent Auto-Scaling
- Predictive scaling based on ML predictions.
- Reactive scaling based on current CPU utilization.
- Scale-out and scale-in recommendations.
- Minimum and maximum instance limits.
- Scaling event history.

### Performance Evaluation
- Predictive vs reactive scaling comparison.
- Estimated operational cost comparison.
- Estimated energy consumption comparison.
- CPU utilization and latency evaluation.
- Overload analysis.

### Live Simulation
- Automatic workload generation.
- Real-time CPU prediction.
- Dynamic scaling recommendations.
- Manual workload testing.
- High-traffic and low-traffic scenarios.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| Scikit-learn | Machine learning model |
| Random Forest Regression | CPU utilization prediction |
| Pandas | Dataset processing |
| NumPy | Numerical computations |
| Plotly | Interactive charts |
| Joblib | Saving and loading the trained model |

## 5. System Architecture

```text
       Workload Dataset
              |
              v
      Data Preprocessing
              |
              v
      Feature Engineering
              |
              v
     Random Forest Model
              |
              v
      Future CPU Prediction
              |
              v
       Scaling Decision
          /       \
         /         \
        v           v
 Predictive       Reactive
  Scaling          Scaling
        \           /
         \         /
          v       v
       Performance Evaluation
              |
              v
       Streamlit Dashboard
```

## 6. Machine Learning Model

The project uses a Random Forest Regression algorithm to predict future CPU utilization.

### Input Features

- Current CPU utilization
- Memory utilization
- Active users
- Requests per second
- Network traffic
- Historical CPU values
- Rolling CPU averages
- Time-based features

### Prediction Target

Future CPU utilization, approximately five minutes ahead in the generated workload dataset.

### Model Performance

| Metric | Result |
|---|---:|
| MAE | 3.1037 |
| RMSE | 5.5324 |
| R² Score | 0.8985 |

The model achieved an R² score of approximately 0.90 on the test dataset.

## 7. Auto-Scaling Logic

The system compares two scaling strategies.

### Predictive Scaling

Uses the machine learning model's predicted CPU utilization to recommend scaling decisions.

### Reactive Scaling

Uses current CPU utilization to make scaling decisions.

### Scaling Thresholds

| Condition | Decision |
|---|---|
| CPU >= 60% | Scale Out |
| CPU <= 30% | Scale In |
| CPU between 30% and 60% | Maintain |

- Initial instances: 2
- Minimum instances: 1
- Maximum instances: 8

## 8. Performance Evaluation Results

The simulation produced the following estimated results:

| Metric | Predictive | Reactive |
|---|---:|---:|
| Average Instances | 1.222 | 1.537 |
| Maximum Instances | 5 | 8 |
| Estimated Cost | ₹101.50 | ₹127.67 |
| Estimated Energy | 2.000 kWh | 2.275 kWh |
| Average Latency | 56.638 ms | 52.591 ms |
| Overload Minutes | 34 | 10 |
| Scale-Out Actions | 4 | 11 |

### Observed Improvements

- Estimated cost savings: 20.50%
- Estimated energy savings: 12.08%

Predictive scaling reduced estimated resource costs and energy consumption in this simulation. However, it showed higher average latency and more overload minutes than reactive scaling. These results highlight areas for future improvement.

**Note:** Cost, energy, latency, and overload values are simulated estimates, not measurements from an actual cloud provider.

## 9. Project Structure

```text
SmartScale-AI/
│
├── app.py
├── generate_dataset.py
├── requirements.txt
│
├── data/
│   └── cloud_workload.csv
│
├── models/
│   └── random_forest_model.pkl
│
└── modules/
    ├── __init__.py
    ├── evaluator.py
    ├── live_simulator.py
    ├── ml_predictor.py
    ├── monitor.py
    ├── resource_manager.py
    └── workload_generator.py
```

## 10. Installation and Execution

### Prerequisites

- Python 3.10 or later
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/eshheet77/SmartScale-AI.git
```

### Step 2: Navigate to the Project

```bash
cd SmartScale-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser, usually at:

```text
http://localhost:8501
```

## 11. Dashboard Sections

The application contains seven main sections:

1. Overview
2. Workload Monitoring
3. ML Predictions
4. Auto Scaling
5. Performance Evaluation
6. Model Information
7. Live Simulation

## 12. Future Scope

- Integration with real cloud platforms such as AWS, Azure, or Google Cloud.
- Advanced time-series forecasting models such as LSTM.
- Improved prediction accuracy during sudden workload spikes.
- More advanced scaling policies with cooldown and stabilization mechanisms.
- Container-based deployment using Docker and Kubernetes.
- Real-time workload monitoring and automated scaling execution.
- Improved latency and overload management.

## 13. Limitations

- The current implementation uses simulated workloads rather than live cloud infrastructure.
- Scaling decisions are recommendations within a local simulation.
- Cost and energy calculations are estimates.
- The current Random Forest model may underestimate sudden workload spikes.

## 14. Conclusion

SmartScale AI demonstrates how machine learning can support intelligent cloud resource management through workload prediction and predictive auto-scaling.

By comparing predictive and reactive scaling strategies, the project demonstrates the potential benefits of proactive resource allocation, while also identifying performance trade-offs that can be improved in future versions.

The project provides a complete local demonstration environment for machine learning-based cloud resource prediction and intelligent auto-scaling.

---

**GitHub Repository:** https://github.com/eshheet77/SmartScale-AI

**Project:** SmartScale AI  
**Domain:** Cloud Computing, Machine Learning, Resource Management
