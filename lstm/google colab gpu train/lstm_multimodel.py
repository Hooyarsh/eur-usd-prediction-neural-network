#Directly trains the model with fixed hyperparameters (learning_rate=0.001, dropout_rate=0.2, epochs=50, batch_size=32).
#Uses EarlyStopping to prevent overfitting.
#Faster to implement but less flexible due to fixed parameters.

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load dataset
data = pd.read_csv('/content/drive/My Drive/EURUSDHistoricalData.csv')
prices = data['Price'].values.reshape(-1, 1)

# Scaling the data
scaler = StandardScaler()
scaled_prices = scaler.fit_transform(prices)

# Create sequences
def create_sequences(data, input_len=40, output_day=5):
    X, y = [], []
    for i in range(len(data) - input_len - output_day + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len + output_day - 1])  # Predict the 5th day ahead
    return np.array(X), np.array(y)

input_len, output_day = 40, 5
X, y = create_sequences(scaled_prices, input_len, output_day)
X = X.reshape(-1, input_len, 1)

# Split data into train, validation, and test sets
train_size = int(0.7 * len(X))
val_size = max(1, int(0.2 * len(X)))
test_size = max(1, len(X) - train_size - val_size)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Define the hybrid model
def create_hybrid_model(learning_rate=0.001, dropout_rate=0.2):
    # Input layer
    input_layer = Input(shape=(input_len, 1))
    
    # Path 1: LSTM -> Points A and B -> Fully Connected
    path1 = LSTM(64, return_sequences=True)(input_layer)
    path1 = Dropout(dropout_rate)(path1)
    point_A = LSTM(32)(path1)  # Point A
    path1_output = Dense(1)(point_A)  # Final output for Path 1

    # Path 2: CNN layers
    path2 = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
    path2 = Conv1D(32, kernel_size=3, activation='relu')(path2)
    path2 = Conv1D(16, kernel_size=3, activation='relu')(path2)

    # Path 2 Split into Paths 3 and 4
    # Path 3: CNN -> LSTM to Point A
    path3 = LSTM(32)(path2)

    # Path 4: CNN -> 3xCNN -> LSTM to Point B
    path4 = Conv1D(16, kernel_size=3, activation='relu')(path2)
    path4 = Conv1D(8, kernel_size=3, activation='relu')(path4)
    path4 = LSTM(32)(path4)

    # Combine Point A and Path 3 output
    merged_A = concatenate([point_A, path3])
    
    # Combine Point B and Path 4 output
    merged_B = concatenate([path1_output, path4])

    # Final Dense Layers to Output
    output_layer = Dense(1)(merged_B)

    # Model Definition
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Create model
model = create_hybrid_model(learning_rate=0.001, dropout_rate=0.2)

# Training with Early Stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

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