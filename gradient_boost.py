import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

# Define and create directory for saving plots
save_dir = r"C:\Users\Adarsh Pradeep\OneDrive\Desktop\Adarsh_Personal\PAPERS_TO_PUBLISH\plotsresults"
os.makedirs(save_dir, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.2)

# Load and preprocess
df = pd.read_csv("wind_paper.csv")
df = df[df['Patv'] > 0].drop(['Tmstamp', 'TurbID'], axis=1)

N_LAGS = 36
for lag in range(1, N_LAGS + 1):
    df[f'Patv_lag_{lag}'] = df['Patv'].shift(lag)
df['Patv_diff'] = df['Patv'].diff()
df['RollingMean'] = df['Patv'].rolling(window=6).mean()
df.dropna(inplace=True)

X = df.drop('Patv', axis=1).values
y = df['Patv'].values.reshape(-1, 1)

X = MinMaxScaler().fit_transform(X)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]

# ESN Class
class ESN:
    def __init__(self, input_dim, reservoir_size=2000, spectral_radius=0.98, sparsity=0.05,
                 reg=1e-3, leaky_rate=0.15, washout=150):
        self.input_dim = input_dim + 1
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.reg = reg
        self.leaky_rate = leaky_rate
        self.washout = washout

        self.Win = np.random.uniform(-0.5, 0.5, (reservoir_size, self.input_dim))
        W = np.random.rand(reservoir_size, reservoir_size) - 0.5
        W[np.random.rand(*W.shape) > sparsity] = 0
        eig_val = max(abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / eig_val)

    def _update(self, state, input_):
        u = np.concatenate((input_, [1]))
        pre_activation = np.dot(self.Win, u) + np.dot(self.W, state)
        return (1 - self.leaky_rate) * state + self.leaky_rate * np.tanh(pre_activation)

    def fit(self, X, y):
        states = []
        state = np.zeros(self.reservoir_size)
        for x in X:
            state = self._update(state, x)
            states.append(state.copy())
        states = np.array(states)
        X_eff, y_eff = states[self.washout:], y[self.washout:]
        self.Wout = Ridge(alpha=self.reg, fit_intercept=False).fit(X_eff, y_eff).coef_.T.flatten()

    def predict(self, X):
        preds, state = [], np.zeros(self.reservoir_size)
        for x in X:
            state = self._update(state, x)
            preds.append(state @ self.Wout)
        return np.array(preds)

# Train & predict
start = time.time()
esn = ESN(input_dim=X.shape[1])
esn.fit(X_train, y_train)
y_pred_scaled = esn.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_actual = y_scaler.inverse_transform(y_test)
end = time.time()

# Metrics
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
r2 = r2_score(y_test_actual, y_pred)
train_time = end - start

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}, TrainTime: {train_time:.2f}s")

# Prepare error and test dataframe
df_test = df.iloc[split:].copy()
df_test['Error'] = abs(df_test['Patv'].values - y_pred.flatten())

# Plot 1: Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:100], label='Actual', color='black')
plt.plot(y_pred[:100], label='Predicted', color='orange')
plt.title("Forecast vs Actual (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Power Output (Patv) [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "forecast_vs_actual.png"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Residual Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_test['Error'], kde=True, color='red')
plt.title("Residual Distribution")
plt.xlabel("Absolute Error (kW)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "residual_distribution.png"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Error Over Time
plt.figure(fig)
