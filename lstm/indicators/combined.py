import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

# Fetch EUR/USD Data from Alpha Vantage API
def fetch_candles(symbol, interval='5min', api_key='4ZQTRVFFDVTKZSNF'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if f"Time Series ({interval})" in data:
            df = pd.DataFrame(data[f"Time Series ({interval})"]).T
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['time'] = pd.to_datetime(df.index)
            df = df.reset_index(drop=True)
            return df[['time', 'close', 'volume']]
        else:
            print("Error: No data found in API response.")
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
    return pd.DataFrame()  # Return empty DataFrame if API fails

# Define Technical Indicators
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    ema_short = data['close'].ewm(span=short_period, adjust=False).mean()
    ema_long = data['close'].ewm(span=long_period, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return upper_band, lower_band

def calculate_vwap(data):
    vwap = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()
    return vwap

def create_lagged_features(data, column, lags=3):
    for lag in range(1, lags + 1):
        data[f'{column}_lag{lag}'] = data[column].shift(lag)
    return data

# Preprocess Dataset
def prepare_lstm_data(data, features, target, timesteps=10):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[features].iloc[i:i + timesteps].values)
        y.append(data[target].iloc[i + timesteps])
    return np.array(X), np.array(y)

# Fetch and Process Data
symbol = 'EURUSD'
data = fetch_candles(symbol)

# Check if data was fetched successfully
if data.empty:
    raise ValueError("No data fetched. Check the API or symbol.")

# Calculate Technical Indicators
data['rsi'] = calculate_rsi(data)
data['macd'], data['signal'] = calculate_macd(data)
data['upper_band'], data['lower_band'] = calculate_bollinger_bands(data)
data['vwap'] = calculate_vwap(data)

# Create Lagged Features
data = create_lagged_features(data, 'close', lags=3)
data = data.dropna()  # Drop rows with NaN values after adding indicators and lags

# Define Features
features = ['rsi', 'macd', 'signal', 'upper_band', 'lower_band', 'vwap', 
            'close_lag1', 'close_lag2', 'close_lag3']

# Debugging: Ensure enough data is available
print(f"Data shape after preprocessing: {data.shape}")
if data[features].shape[0] == 0:
    raise ValueError("Not enough data after preprocessing. Please check the data source and preprocessing steps.")

# Scale Data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Prepare LSTM Inputs
timesteps = 10
X, y = prepare_lstm_data(data, features, 'close', timesteps=timesteps)

# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM Model
def create_model(learning_rate=0.001, neurons=50, dropout_rate=0.2, optimizer='adam', activation='relu'):
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=(timesteps, len(features))),
        Dropout(dropout_rate),
        LSTM(neurons, return_sequences=False),
        Dropout(dropout_rate),
        Dense(neurons, activation=activation),
        Dense(1)
    ])
    opt = Adam(learning_rate=learning_rate) if optimizer == 'adam' else RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse')
    return model

# Wrap the model for GridSearchCV
model = KerasRegressor(build_fn=create_model)

# Hyperparameter Grid
param_grid = {
    'learning_rate': [0.001, 0.01],
    'neurons': [32, 64],
    'dropout_rate': [0.2, 0.3],
    'batch_size': [16, 32],
    'epochs': [20, 50],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop']
}

# Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Best Parameters
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
best_params = grid_result.best_params_

# Final Model with Best Parameters
final_model = create_model(
    learning_rate=best_params['learning_rate'],
    neurons=best_params['neurons'],
    dropout_rate=best_params['dropout_rate'],
    activation=best_params['activation'],
    optimizer=best_params['optimizer']
)

# Train the Final Model
history = final_model.fit(
    X_train, y_train, 
    epochs=best_params['epochs'], 
    batch_size=best_params['batch_size'], 
    validation_data=(X_test, y_test)
)

# Evaluate the Model
y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R2: {r2}')

# Plot Predictions
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label='Actual Prices', color='blue')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Prices', color='red', linestyle='dashed')
plt.title('True vs Predicted Prices')
plt.legend()
plt.show()