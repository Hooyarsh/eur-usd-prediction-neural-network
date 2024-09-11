import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('EURUSDHistoricalData.csv')

print(f"Data loaded: {data.shape[0]} rows and {data.shape[1]} columns")

prices = data['Price'].values.reshape(-1, 1)

print(f"Prices data: {prices.shape[0]} data points")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

def create_sequences(data, input_len=40):
    X = []
    y = []
    for i in range(len(data) - input_len):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len])
    return np.array(X), np.array(y)

input_len = 40
X, y = create_sequences(scaled_prices, input_len)

print(f"X shape: {X.shape}, y shape: {y.shape}")

train_size = int(0.7 * len(X))
val_size = max(1, int(0.2 * len(X)))
test_size = max(1, len(X) - train_size - val_size)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

class FCN(nn.Module):
    def __init__(self, input_len):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_len * 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = FCN(input_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
batch_size = 32

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_losses.append(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            if len(X_val) > 0:
                val_predictions = model(X_val)
                val_loss = criterion(val_predictions, y_val)
            else:
                val_loss = float('nan')

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

def plot_predictions(y_val, predictions, scaler):
    y_val_inverse = scaler.inverse_transform(y_val.cpu().numpy())
    predictions_inverse = scaler.inverse_transform(predictions.cpu().numpy())

    print("y_val_inverse shape:", y_val_inverse.shape)
    print("predictions_inverse shape:", predictions_inverse.shape)
    print("Sample y_val_inverse:", y_val_inverse[:5])
    print("Sample predictions_inverse:", predictions_inverse[:5])

    plt.figure(figsize=(10, 5))
    plt.plot(y_val_inverse, label='Actual Prices', color='b')
    plt.plot(predictions_inverse, label='Predicted Prices', color='r', linestyle='dashed')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.title('Predicted vs Actual Prices')
    plt.legend()
    plt.show()

train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)

model.eval()
with torch.no_grad():
    if len(X_val) > 0:
        val_predictions = model(X_val)

if len(X_val) > 0:
    plot_predictions(y_val, val_predictions, scaler)

with torch.no_grad():
    last_40_days = X_test[-1].unsqueeze(0)
    next_day_prediction = model(last_40_days)
    next_day_price = scaler.inverse_transform(next_day_prediction.numpy())
    print("Predicted price for the next day:", next_day_price[0][0])

with torch.no_grad():
    test_predictions = model(X_test)

plot_predictions(y_test, test_predictions, scaler)


