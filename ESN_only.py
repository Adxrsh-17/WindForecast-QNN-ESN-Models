import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

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

# Plots

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

plt.figure(figsize=(8, 5))
sns.histplot(df_test['Error'], kde=True, color='red')
plt.title("Residual Distribution")
plt.xlabel("Absolute Error (kW)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df_test['Error'][:300], label='Error', color='purple')
plt.title("Prediction Error Over Time")
plt.xlabel("Sample Index")
plt.ylabel("Absolute Error (kW)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.title("Scatter Plot: Actual vs Predicted")
plt.xlabel("Actual Power Output (kW)")
plt.ylabel("Predicted Power Output (kW)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
ax = plt.gca()
plot_acf(df['Patv'], lags=30, ax=ax)
plt.title("Lag Correlation Plot (Autocorrelation of Power Output)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Fix future warning by using hue=x with legend=False
df_test['Month'] = pd.to_datetime(df_test.index, errors='coerce').month
month_order = [1,2,3,4,5,6,7,8,9,10,11,12]
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(14, 7))
sns.boxplot(x='Month', y='Error', data=df_test, hue='Month', palette="Set2", 
            order=month_order, dodge=False, legend=False)
plt.xticks(ticks=range(12), labels=month_names, rotation=45)
plt.title("Error Box Plot by Month")
plt.xlabel("Month")
plt.ylabel("Absolute Error (kW)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

chunks = len(df_test) // 500
residuals_trimmed = df_test['Error'].values[:chunks*500]

plt.figure(figsize=(12, 7))
sns.boxplot(data=pd.DataFrame(residuals_trimmed.reshape(-1, 500)).T, color='skyblue')
plt.title("Box Plot of Errors (per 500 samples)")
plt.xlabel("Chunk Index")
plt.ylabel("Absolute Error (kW)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

feature_importance = np.abs(esn.Wout)
plt.figure(figsize=(10, 5))
plt.bar(range(len(feature_importance)), feature_importance, color='seagreen')
plt.title("Feature Importance (Ridge Coef Magnitude)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

decomp = seasonal_decompose(df['Patv'].values[:500], model='additive', period=24)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.suptitle("Time-Series Decomposition of Power Output (Patv)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
