# use gridsearchcv for hyperoptimization - 2 layer lstm+FullyConnected


from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load dataset
data = pd.read_csv('/content/drive/My Drive/EURUSDHistoricalData.csv')
prices = data['Price'].values.reshape(-1, 1)

# Scaling the data
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
scaled_prices = scaler.fit_transform(prices)

# Function to create sequences
def create_sequences(data, input_len=40, output_day=5):
    X = []
    y = []
    for i in range(len(data) - input_len - output_day + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len + output_day - 1])  # 5th day ahead
    return np.array(X), np.array(y)

input_len = 40
output_day = 5
X, y = create_sequences(scaled_prices, input_len, output_day)
X = X.reshape(-1, input_len, 1)

# Splitting data into training, validation, and test sets
train_size = int(0.7 * len(X))
val_size = max(1, int(0.2 * len(X)))
test_size = max(1, len(X) - train_size - val_size)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Function to create the model
def create_model(learning_rate=0.001, neurons=64, dropout_rate=0.2, optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(input_len, 1)))  # First LSTM layer
    model.add(Dropout(dropout_rate))  # Dropout regularization
    model.add(LSTM(neurons))  # Second LSTM layer
    model.add(Dense(neurons, activation=activation))  # Fully connected layer
    model.add(Dense(1))  # Final output layer (one value for 5th day)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    return model

# Wrap the model for GridSearchCV
model = KerasRegressor(build_fn=create_model)

# Define the grid search parameters
param_grid = {
    'learning_rate': [0.001, 0.01],
    'neurons': [32, 64],
    'dropout_rate': [0.2, 0.3],
    'batch_size': [16, 32],
    'epochs': [50, 100],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop']
}

# Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Output best parameters
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Use best parameters to create the final model
best_params = grid_result.best_params_
final_model = create_model(
    learning_rate=best_params['learning_rate'],
    neurons=best_params['neurons'],
    dropout_rate=best_params['dropout_rate'],
    activation=best_params['activation'],
    optimizer=best_params['optimizer']
)

# Train the model using the best hyperparameters
history = final_model.fit(
    X_train, y_train, 
    epochs=best_params['epochs'], 
    batch_size=best_params['batch_size'], 
    validation_data=(X_val, y_val)
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
test_predictions = final_model.predict(X_test)
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
fifth_day_prediction = final_model.predict(last_40_days)
fifth_day_price = scaler.inverse_transform(fifth_day_prediction)
print("Predicted price for the 5th day ahead:", fifth_day_price)