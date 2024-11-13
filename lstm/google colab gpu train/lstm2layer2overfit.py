from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Load the dataset
data = pd.read_csv('/content/drive/My Drive/EURUSDHistoricalData.csv')
prices = data['Price'].values.reshape(-1, 1)

# Scaling the data
scaler = StandardScaler()
scaled_prices = scaler.fit_transform(prices)

# Create sequences with an input window of 40 and predict only the 5th day ahead
def create_sequences(data, input_len=40, output_day=5):
    X = []
    y = []
    for i in range(len(data) - input_len - output_day + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len + output_day - 1])  # Select only the 5th day
    return np.array(X), np.array(y)

input_len = 40
output_day = 5
X, y = create_sequences(scaled_prices, input_len, output_day)

# Reshape the data for LSTM: (batch_size, sequence_length, features)
X = X.reshape(-1, input_len, 1)

# Adjusted split: train 80%, validation 10%, test 10%
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Hyperparameters: reduce learning rate, tune dropout, batch size, epochs
learning_rate = 0.0005
dropout_rate = 0.3
batch_size = 32
epochs = 50

# Function to create and compile the LSTM model with L2 regularization and dropout
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(input_len, 1), kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_regularizer=l2(0.001)))  # Output layer

    # Compile with a lower learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with regularization and early stopping
model = create_lstm_model()
history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size,
    validation_data=(X_val, y_val), callbacks=[early_stopping]
)

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Testing the model
test_predictions = model.predict(X_test)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
test_predictions_inverse = scaler.inverse_transform(test_predictions)

# Plot the predictions vs actual prices
plt.plot(y_test_inverse, label='Actual 5th Day Prices', color='b')
plt.plot(test_predictions_inverse, label='Predicted 5th Day Prices', linestyle='dashed', color='r')
plt.title('Predicted vs Actual 5th Day Prices on Test Set')
plt.legend()
plt.show()

# Predict the 5th day ahead using the last 40 days of test data
last_40_days = X_test[-1].reshape(1, input_len, 1)
fifth_day_prediction = model.predict(last_40_days)
fifth_day_price = scaler.inverse_transform(fifth_day_prediction)
print("Predicted price for the 5th day ahead:", fifth_day_price)