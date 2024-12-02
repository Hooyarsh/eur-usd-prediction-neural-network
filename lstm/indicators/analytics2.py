#using api to load real-time data

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define technical indicator calculations
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

def create_lagged_features(data, column, lags=3):
    for lag in range(1, lags + 1):
        data[f'{column}_lag{lag}'] = data[column].shift(lag)
    return data

# Fetch data from Alpha Vantage API
def fetch_candles(symbol, interval='5min', api_key='4ZQTRVFFDVTKZSNF'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "Time Series (5min)" in data:
            df = pd.DataFrame(data["Time Series (5min)"]).T
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['time'] = pd.to_datetime(df.index)
            df.reset_index(drop=True, inplace=True)
            return df[['time', 'close', 'volume']]
        else:
            print("Error: No data found in API response.")
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
    return pd.DataFrame()  # Return empty DataFrame if API fails

# Fetch and preprocess the data
symbol = 'EURUSD'
data = fetch_candles(symbol)

# Check if data was fetched successfully
if data.empty:
    print("No data fetched. Check the API or symbol.")
else:
    # Calculate technical indicators
    data['rsi'] = calculate_rsi(data)
    data['macd'], data['signal'] = calculate_macd(data)
    data['upper_band'], data['lower_band'] = calculate_bollinger_bands(data)
    data = create_lagged_features(data, 'close')

    # Debug: Check for NaNs and data size
    print("Data shape before dropping NaNs:", data.shape)
    print("NaN values in the features:\n", data.isnull().sum())

    # Drop rows with NaN values
    data = data.dropna()

    if data.empty:
        print("Insufficient data after computing technical indicators. Try fetching more data or check the parameters.")
    else:
        print("Data shape after dropping NaNs:", data.shape)

        # Define features
        features = ['rsi', 'macd', 'signal', 'upper_band', 'lower_band', 
                    'close_lag1', 'close_lag2', 'close_lag3']

        # Scale features
        scaler = MinMaxScaler()
        data[features] = scaler.fit_transform(data[features])

        # Prepare LSTM Inputs
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        X_train = np.array([train_data[features].iloc[i:i+10].values for i in range(len(train_data) - 10)])
        y_train = train_data['close'].iloc[10:].values
        X_test = np.array([test_data[features].iloc[i:i+10].values for i in range(len(test_data) - 10)])
        y_test = test_data['close'].iloc[10:].values

        # Build LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

        # Evaluate the model
        y_pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'MSE: {mse}, R2: {r2}')

        # Plot predictions
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(y_test)), y_test, label='True Prices', color='blue')
        plt.plot(range(len(y_pred)), y_pred, label='Predicted Prices', color='red')
        plt.title("True vs Predicted Prices")
        plt.legend()
        plt.show()