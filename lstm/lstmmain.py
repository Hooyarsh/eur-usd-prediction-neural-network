#lstm model with pytorch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('EURUSDHistoricalData.csv')
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

# Converting the data into PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)  # Single day output, reshape to (batch_size, 1)
y_val = torch.FloatTensor(y_val).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Final layer outputs the 5th day price
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM layer outputs (all hidden states, final cell state)
        last_hidden_state = lstm_out[:, -1, :]  # Use only the last hidden state for prediction
        out = self.fc(last_hidden_state)  # Pass it through the fully connected layer
        return out

# Loss with regularization function
def loss_with_regularization(predictions, targets, model, l1_lambda=0.002, l2_lambda=0.002):
    mse_loss = nn.MSELoss()(predictions, targets)
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    total_loss = mse_loss + l1_lambda * l1_norm + l2_lambda * l2_norm
    return total_loss

# Training function for the LSTM model
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.002, l2_lambda=0.002):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            
            predictions = model(batch_X)
            loss = loss_with_regularization(predictions, batch_y, model, l1_lambda, l2_lambda)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = loss_with_regularization(val_predictions, y_val, model, l1_lambda, l2_lambda)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return val_predictions

# Plotting the predictions vs actual prices
def plot_predictions(y_val, predictions, scaler, label):
    y_val_inverse = scaler.inverse_transform(y_val.cpu().numpy())
    predictions_inverse = scaler.inverse_transform(predictions.cpu().numpy())
    
    plt.plot(y_val_inverse, label=f'Actual 5th Day Prices {label}', color='b')
    plt.plot(predictions_inverse, label=f'Predicted 5th Day Prices {label}', linestyle='dashed', color='r')
    plt.title(f'Predicted vs Actual 5th Day Prices for {label}')
    plt.legend()
    plt.show()

# Initialize LSTM model
input_size = 1  # We have only one feature (price)
hidden_size = 64  # Number of LSTM units in each layer
num_layers = 2    # Number of LSTM layers

lstm_model = LSTMModel(input_size, hidden_size, num_layers)

# Train the model
epochs = 100
batch_size = 32
val_predictions_lstm = train_model(lstm_model, X_train, y_train, X_val, y_val, epochs, batch_size)

# Plot the validation predictions
plot_predictions(y_val, val_predictions_lstm, scaler, "5th Day LSTM Prediction")

# Test with the last 40 days of test data
with torch.no_grad():
    last_40_days = X_test[-1].unsqueeze(0)
    fifth_day_prediction = lstm_model(last_40_days)
    fifth_day_price = scaler.inverse_transform(fifth_day_prediction.numpy())
    print("Predicted price for the 5th day ahead:", fifth_day_price)