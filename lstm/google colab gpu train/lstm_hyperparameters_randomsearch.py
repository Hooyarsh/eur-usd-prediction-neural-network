from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
import tensorflow as tf

# Load the dataset
data = pd.read_csv('/content/drive/My Drive/EURUSDHistoricalData.csv')
prices = data['Price'].values.reshape(-1, 1)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
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

# Splitting the data into training, validation, and test sets
train_size = int(0.7 * len(X))
val_size = max(1, int(0.2 * len(X)))
test_size = max(1, len(X) - train_size - val_size)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Check if GPU is available
device_name = tf.config.list_physical_devices('GPU')
if device_name:
    print(f"GPU is available: {device_name}")
else:
    print("GPU is not available. Using CPU.")

# Define a hypermodel
def build_model(hp):
    model = Sequential()
    # Tuning the number of units in LSTM
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32), return_sequences=False, input_shape=(input_len, 1)))
    
    # Adding Dense layer with 1 unit (predicting one price)
    model.add(Dense(1))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error'
    )
    return model

# Hyperparameter tuning using Random Search
tuner = RandomSearch(
    build_model,
    objective='val_loss',  # We want to minimize validation loss
    max_trials=5,          # Number of different models to try
    executions_per_trial=1, # Number of times to train each model
    directory='tuning_dir', # Directory where to store logs and checkpoints
    project_name='lstm_tuning'
)

# Search space for batch size
batch_size = [32, 64, 128]

# Run the tuner search to find the best hyperparameters
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the LSTM layer is {best_hps.get('units')},
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)

# Train the model with the best batch size
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

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
last_40_days = X_test[-1].reshape(1, input_len, 1)  # Reshape for the model
fifth_day_prediction = model.predict(last_40_days)
fifth_day_price = scaler.inverse_transform(fifth_day_prediction)
print("Predicted price for the 5th day ahead:", fifth_day_price)