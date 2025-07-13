# 🌬️ Wind Power Forecasting: QNN, ESN & Hybrid Models

A comparative analysis of machine learning models for wind power forecasting using SCADA-based time-series data. This repository includes:

- ⚛️ **Quantum Neural Network (QNN)** using PennyLane & TensorFlow
- 💧 **Echo State Network (ESN)** for efficient temporal modeling
- ⚙️ **Hybrid ESN + DNN** architecture for nonlinear sequence learning
- 📊 **Gradient Boosting (GB)** as a baseline ensemble regressor

---

## 🧠 Motivation

Accurate short-term wind power forecasting is critical for grid reliability, energy trading, and turbine control. Traditional models often struggle with the nonlinearity and temporal dependencies in wind data. This project benchmarks multiple paradigms—including quantum-inspired learning and reservoir computing—to identify the best-performing model.

---

## 📁 Repository Contents

| File Name         | Description |
|------------------|-------------|
| `q.py`           | Quantum Neural Network using PennyLane + Keras |
| `ESN_only.py`    | Standalone Echo State Network (ESN) |
| `esn_dense.py`   | Hybrid model combining ESN and Dense layers |
| `gradient_boost.py` | Gradient Boosting model with lag features and ESN |

---

## 📊 Dataset

- **Source:** SCADA time-series data from industrial wind turbines
- **Target Variable:** `Patv` (Actual Power Output in kW)
- **Features:** Wind speed, direction, temperature, operational metrics
- **Preprocessing:**
  - Removed invalid entries where `Patv <= 0`
  - Created 36 lag features + rolling mean + differencing
  - Min-max normalization applied
  - 80/20 split for train/test

---

## 🧪 Models & Methodology

### 🔹 Gradient Boosting (GB)
- Tree-based ensemble model
- Pros: Fast, robust to noise
- Cons: Poor at modeling temporal dependencies

### ⚛️ Quantum Neural Network (QNN)
- Simulated on classical backend using PennyLane
- Pros: Theoretically powerful due to quantum entanglement
- Cons: Poor performance due to barren plateaus and NISQ limitations

### 💧 Echo State Network (ESN)
- Reservoir computing model with fast training
- Pros: Strong temporal memory, lightweight
- Cons: Limited adaptability due to fixed reservoir weights

### ⚙️ Hybrid ESN + DNN
- Combines ESN for memory with DNN for nonlinearity
- Pros: Best generalization and predictive accuracy

---

## 📈 Results Summary

| Model               | RMSE   | MAE   | MAPE   | R²     | Train Time |
|--------------------|--------|-------|--------|--------|------------|
| **Hybrid ESN+DNN** | 25.83  | 13.76 | 10.64% | 0.9954 | 91.3 s     |
| **ESN**            | 40.30  | 10.39 | 6.88%  | 0.9889 | 50.9 s     |
| Gradient Boosting  | 101.57 | 67.14 | 35.82% | 0.9295 | 42.0 s     |
| QNN (Simulated)    | 140.35 | 93.87 | 43.86% | 0.8869 | N/A        |

---

## 🔍 Key Findings

- ✅ **Best Model:** Hybrid ESN+DNN — excels at combining temporal memory with nonlinear learning.
- ⚡ **Fastest Training:** ESN — trains in under 1 minute while achieving high accuracy.
- ❌ **QNN Limitations:** Barren plateau phenomenon and NISQ constraints degrade performance significantly.
- ⚠️ **Gradient Boosting:** Good for baseline but inadequate for time-series without temporal memory.

---

## 📊 Visualizations

Each script generates the following plots:

- Forecast vs Actual (first 100 samples)
- Residual Distribution
- Error over Time
- Scatter Plot of Actual vs Predicted
- Feature Importance (for Ridge/ESN)
- Time-Series Decomposition (optional)


