#two sequential regression layers
#Uses GridSearchCV to perform a hyperparameter search over learning_rate, dropout_rate, batch_size, and epochs.
#Selects the best combination of hyperparameters based on cross-validation.
#Implements a full grid search with cross-validation to identify the best hyperparameters for the model.
#More flexible and likely to yield better results by tuning hyperparameters through GridSearchCV.



from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Flatten, concatenate
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('/content/drive/My Drive/EURUSDHistoricalData.csv')
prices = data['Price'].values.reshape(-1, 1)

# Scaling the data
scaler = StandardScaler()
scaled_prices = scaler.fit_transform(prices)

# Create sequences function
def create_sequences(data, input_len=40, output_day=5):
    X, y = [], []
    for i in range(len(data) - input_len - output_day + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len + output_day - 1])  # Predict the 5th day ahead
    return np.array(X), np.array(y)

input_len = 40
output_day = 5
X, y = create_sequences(scaled_prices, input_len, output_day)
X = X.reshape(-1, input_len, 1)

# Split data into training, validation, and test sets
train_size = int(0.7 * len(X))
val_size = max(1, int(0.2 * len(X)))
test_size = max(1, len(X) - train_size - val_size)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Function to create the hybrid multimodel
def create_hybrid_model(learning_rate=0.001, dropout_rate=0.2):
    input_layer = Input(shape=(input_len, 1))

    # Path 1: LSTM layer
    x1 = LSTM(64, return_sequences=True)(input_layer)
    x1 = LSTM(32)(x1)
    point_A = Dense(32, activation='relu')(x1)

    # Path 2: CNN layers
    x2 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    x2 = Conv1D(filters=32, kernel_size=3, activation='relu')(x2)
    x2 = Conv1D(filters=16, kernel_size=3, activation='relu')(x2)
    x2_flat = Flatten()(x2)

    # Path 3: LSTM after CNNs, ending at Point A
    path3 = LSTM(32)(x2_flat)
    path3_output = Dense(32, activation='relu')(path3)

    # Path 4: Additional CNN and LSTM layers, ending at Point B
    x4 = Conv1D(filters=16, kernel_size=3, activation='relu')(x2)
    x4 = Conv1D(filters=8, kernel_size=3, activation='relu')(x4)
    x4 = Flatten()(x4)
    path4 = LSTM(32)(x4)
    point_B = Dense(32, activation='relu')(path4)

    # Merge points A, B, and Path 3
    merged = concatenate([point_A, path3_output, point_B])

    # Sequential Linear Regression Layers
    regression_layer_1 = Dense(1)(merged)  # First linear regression layer
    regression_layer_2 = Dense(1)(regression_layer_1)  # Second linear regression layer

    # Define and compile the model
    model = Model(inputs=input_layer, outputs=regression_layer_2)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    return model

# Wrap the model for GridSearchCV
model = KerasRegressor(build_fn=create_hybrid_model)

# Define the grid search parameters
param_grid = {
    'learning_rate': [0.001, 0.01],
    'dropout_rate': [0.2, 0.3],
    'batch_size': [16, 32],
    'epochs': [50, 100]
}

# Grid Search with cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Output best parameters
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Use best parameters to create the final model
best_params = grid_result.best_params_
final_model = create_hybrid_model(
    learning_rate=best_params['learning_rate'],
    dropout_rate=best_params['dropout_rate']
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
last_40_days = X_test[-1].reshape(1, input_len, 1)
fifth_day_prediction = final_model.predict(last_40_days)
fifth_day_price = scaler.inverse_transform(fifth_day_prediction)
print("Predicted price for the 5th day ahead:", fifth_day_price)