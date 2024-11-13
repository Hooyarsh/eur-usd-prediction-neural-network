#    1.  Learning rate: Reduced to 0.0001 and 0.001.
#    2.  Epochs: Reduced to a range of [30, 50].
#    3.  Dropout rate: Increased to [0.3, 0.5].
#    4.  Train/Test Split: 80% training, 15% validation, and 5% testing.
#    5.  Batch size: Increased to [32, 64].
#    6.  L2 Regularization: Introduced with regularization values of 0.01 and 0.001.



from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Load dataset
data = pd.read_csv('/content/drive/My Drive/EURUSDHistoricalData.csv')
prices = data['Price'].values.reshape(-1, 1)

# Scaling the data
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
train_size = int(0.8 * len(X))  # 80% for training
val_size = max(1, int(0.15 * len(X)))  # 15% for validation
test_size = max(1, len(X) - train_size - val_size)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Function to create the model
def create_model(learning_rate=0.001, neurons=64, dropout_rate=0.2, optimizer='adam', activation='relu', regularization=0.01):
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(input_len, 1), 
                   kernel_regularizer=l2(regularization)))  # First LSTM with L2 regularization
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, kernel_regularizer=l2(regularization)))  # Second LSTM with L2 regularization
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))  # Final output layer
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = RMSprop(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    return model

# Wrap the model for GridSearchCV
model = KerasRegressor(build_fn=create_model)

# Define the grid search parameters
param_grid = {
    'learning_rate': [0.0001, 0.001],  # Decreased learning rate
    'neurons': [32, 64],
    'dropout_rate': [0.3, 0.5],  # Increased dropout rate
    'batch_size': [32, 64],  # Increased batch size
    'epochs': [30, 50],  # Reduced epochs
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop'],
    'regularization': [0.01, 0.001]  # L2 regularization
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
    optimizer=best_params['optimizer'],
    regularization=best_params['regularization']
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