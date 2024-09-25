import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('EURUSDHistoricalData.csv')
data.info()
prices = data['Price'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

def create_sequences(data, input_len, output_len):
    X = []
    y = []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:(i + input_len)])
        y.append(data[(i + input_len):(i + input_len + output_len)])
    return np.array(X), np.array(y)

input_len = 40
output_len = 5
X, y = create_sequences(scaled_prices, input_len, output_len)

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(input_len, 1)))
model.add(Dense(output_len))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))

predictions = model.predict(X_val)

predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, output_len)
actual_values = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1, output_len)

print("Predicted prices for the next 5 days:", predictions[0])
print("Actual prices for the next 5 days:", actual_values[0])

