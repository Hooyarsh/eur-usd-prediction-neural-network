# use manual grid search - 2 layer lstm
# No module named 'keras.wrappers'


from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the dataset
data = pd.read_csv('/content/drive/My Drive/EURUSDHistoricalData.csv')
prices = data['Price'].values.reshape(-1, 1)

# Scaling the data
scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
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

# Hyperparameter search space
batch_sizes = [32, 64]
epochs_list = [50, 100]
learning_rates = [0.001, 0.01]
dropouts = [0.2, 0.3]

# Function to create and compile the LSTM model
def create_lstm_model(learning_rate=0.001, dropout_rate=0.2):
    model = Sequential()
    # First LSTM layer
    model.add(LSTM(units=64, return_sequences=True, input_shape=(input_len, 1)))
    model.add(Dropout(dropout_rate))  # Apply dropout
    # Second LSTM layer
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(dropout_rate))  # Apply dropout
    # Fully connected layer (dense)
    model.add(Dense(1))  # Output one value, the price for the 5th day ahead

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

# Perform grid search manually
best_val_loss = float('inf')
best_params = {}

for batch_size in batch_sizes:
    for epochs in epochs_list:
        for lr in learning_rates:
            for dropout_rate in dropouts:
                print(f"Training model with batch_size={batch_size}, epochs={epochs}, lr={lr}, dropout={dropout_rate}")
                model = create_lstm_model(learning_rate=lr, dropout_rate=dropout_rate)
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
                
                # Get validation loss to evaluate model
                val_loss = history.history['val_loss'][-1]
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'learning_rate': lr,
                        'dropout_rate': dropout_rate
                    }
                    print(f"New best model found: val_loss={val_loss}, params={best_params}")

# Display the best parameters
print(f"Best hyperparameters: {best_params}")

# Now, train the model using the best parameters
model = create_lstm_model(learning_rate=best_params['learning_rate'], dropout_rate=best_params['dropout_rate'])
history = model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(X_val, y_val))

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