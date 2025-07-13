import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

sns.set(style="whitegrid", font_scale=1.2)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Avoid GPU-related crashes

# ==== Load & Preprocess Data ====
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

# ==== Echo State Network (ESN) ====
class ESN:
    def __init__(self, input_dim, reservoir_size=900, spectral_radius=1.0, sparsity=0.05,
                 reg=1e-5, leaky_rate=0.3, washout=60):
        self.input_dim = input_dim + 1  # +1 for bias
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
        u = np.concatenate((input_, [1]))  # Add bias
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
        self.trained_states = states  # For hybrid

    def predict(self, X):
        preds, state = [], np.zeros(self.reservoir_size)
        for x in X:
            state = self._update(state, x)
            preds.append(state @ self.Wout)
        return np.array(preds)

# ==== Hybrid Dense Network ====
def build_hybrid(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ==== Train ESN ====
start = time.time()
esn = ESN(input_dim=X.shape[1])
esn.fit(X_train, y_train)
esn_pred_train = esn.predict(X_train)
esn_pred_test = esn.predict(X_test)

# ==== Hybrid Training ====
X_hybrid_train = esn_pred_train.reshape(-1, 1)
X_hybrid_test = esn_pred_test.reshape(-1, 1)

hybrid_model = build_hybrid(1)
hybrid_model.fit(X_hybrid_train, y_train, epochs=50, batch_size=32, verbose=0)

y_pred_scaled = hybrid_model.predict(X_hybrid_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_actual = y_scaler.inverse_transform(y_test)
end = time.time()

# ==== Metrics ====
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
r2 = r2_score(y_test_actual, y_pred)
train_time = end - start

print("Hybrid Model Metrics:")
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}, Train Time: {train_time:.2f}s")

# ==== Plot Actual vs Predicted ====
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:100], label='Actual', color='black')
plt.plot(y_pred[:100], label='Predicted', color='orange')
plt.title("Forecast vs Actual (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Power Output (Patv) [kW]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
