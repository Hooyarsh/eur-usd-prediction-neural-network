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

def loss_with_regularization(predictions, targets, model, l1_lambda=0.0, l2_lambda=0.0):
    mse_loss = nn.MSELoss()(predictions, targets)

    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    
    total_loss = mse_loss + l1_lambda * l1_norm + l2_lambda * l2_norm
    return total_loss


model = FCN(input_len)
criterion = nn.MSELoss()

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.0, l2_lambda=0.0):
    train_losses = []
    val_losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            predictions = model(batch_X)
            loss = loss_with_regularization(predictions, batch_y, model, l1_lambda, l2_lambda)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = loss_with_regularization(val_predictions, y_val, model, l1_lambda, l2_lambda)

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return val_predictions

def plot_predictions(y_val, predictions, scaler, label):
    y_val_inverse = scaler.inverse_transform(y_val.cpu().numpy())
    predictions_inverse = scaler.inverse_transform(predictions.cpu().numpy())

    plt.plot(y_val_inverse, label=f'Actual Prices ({label})', color='b')
    plt.plot(predictions_inverse, label=f'Predicted Prices ({label})', linestyle='dashed')

epochs = 100
batch_size = 32

#Model without regularization
model_wo_reg = FCN(input_len)
val_predictions_wo_reg = train_model(model_wo_reg, X_train, y_train, X_val, y_val, epochs, batch_size)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_wo_reg, scaler, "No Regularization")
plt.title('Predicted vs Actual Prices (No Regularization)')
plt.legend()
plt.show()

#Model with L1=0.001, L2=0.001
model_l1_l2_01 = FCN(input_len)
val_predictions_l1_l2_01 = train_model(model_l1_l2_01, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.002, l2_lambda=0.002)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_l1_l2_01, scaler, "L1=0.00, L2=0.001")
plt.title('Predicted vs Actual Prices (L1=0.001, L2=0.001)')
plt.legend()
plt.show()

#Model with L1=0.002, L2=0.0
model_l1_02 = FCN(input_len)
val_predictions_l1_02 = train_model(model_l1_02, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.004, l2_lambda=0.0)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_l1_02, scaler, "L1=0.002, L2=0.0")
plt.title('Predicted vs Actual Prices (L1=0.002, L2=0.0)')
plt.legend()
plt.show()

#Model with L1=0.0, L2=0.002
model_l2_02 = FCN(input_len)
val_predictions_l2_02 = train_model(model_l2_02, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.0, l2_lambda=0.004)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_l2_02, scaler, "L1=0.0, L2=0.002")
plt.title('Predicted vs Actual Prices (L1=0.0, L2=0.002)')
plt.legend()
plt.show()

# 5. Combined plot
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_wo_reg, scaler, "No Regularization")
plot_predictions(y_val, val_predictions_l1_l2_01, scaler, "L1=0.1, L2=0.1")
plot_predictions(y_val, val_predictions_l1_02, scaler, "L1=0.2, L2=0.0")
plot_predictions(y_val, val_predictions_l2_02, scaler, "L1=0.0, L2=0.2")
plt.title('Combined: Predicted vs Actual Prices')
plt.legend()
plt.show()

def plot_predictions(y_val, predictions, scaler):
    y_val_inverse = scaler.inverse_transform(y_val.cpu().numpy())
    predictions_inverse = scaler.inverse_transform(predictions.cpu().numpy())

    print("y_val_inverse shape:", y_val_inverse.shape)
    print("predictions_inverse shape:", predictions_inverse.shape)
    print("Sample y_val_inverse:", y_val_inverse[:5])
    print("Sample predictions_inverse:", predictions_inverse[:5])

train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)

model.eval()
with torch.no_grad():
    if len(X_val) > 0:
        val_predictions = model(X_val)
        plot_predictions(y_val, val_predictions, scaler)

with torch.no_grad():
    if len(X_test) > 0:
        test_predictions = model(X_test)
        plot_predictions(y_test, test_predictions, scaler)

    last_40_days = X_test[-1].unsqueeze(0)
    next_day_prediction = model(last_40_days)
    next_day_price = scaler.inverse_transform(next_day_prediction.numpy())
    print("Predicted price for the next day:", next_day_price[0][0])