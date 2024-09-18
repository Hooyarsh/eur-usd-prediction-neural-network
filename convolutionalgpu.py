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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=0, padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (input_len // 2), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = CNN2D().to(device)
criterion = nn.MSELoss()

def loss_with_regularization(predictions, targets, model, l1_lambda=0.0, l2_lambda=0.0):
    mse_loss = nn.MSELoss()(predictions, targets)

    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    
    total_loss = mse_loss + l1_lambda * l1_norm + l2_lambda * l2_norm
    return total_loss

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.0, l2_lambda=0.0):
    train_losses = []
    val_losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            predictions = model(batch_X)
            loss = loss_with_regularization(predictions, batch_y, model, l1_lambda, l2_lambda)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val.to(device))
            val_loss = loss_with_regularization(val_predictions, y_val.to(device), model, l1_lambda, l2_lambda)

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

model_wo_reg = CNN2D().to(device)
val_predictions_wo_reg = train_model(model_wo_reg, X_train, y_train, X_val, y_val, epochs, batch_size)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_wo_reg, scaler, "No Regularization")
plt.title('Predicted vs Actual Prices (No Regularization)')
plt.legend()
plt.show()

model_l1_l2_01 = CNN2D().to(device)
val_predictions_l1_l2_01 = train_model(model_l1_l2_01, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.002, l2_lambda=0.002)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_l1_l2_01, scaler, "L1=0.001, L2=0.001")
plt.title('Predicted vs Actual Prices (L1=0.001, L2=0.001)')
plt.legend()
plt.show()

model_l1_02 = CNN2D().to(device)
val_predictions_l1_02 = train_model(model_l1_02, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.004, l2_lambda=0.0)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_l1_02, scaler, "L1=0.002, L2=0.0")
plt.title('Predicted vs Actual Prices (L1=0.002, L2=0.0)')
plt.legend()
plt.show()

model_l2_02 = CNN2D().to(device)
val_predictions_l2_02 = train_model(model_l2_02, X_train, y_train, X_val, y_val, epochs, batch_size, l1_lambda=0.0, l2_lambda=0.004)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_l2_02, scaler, "L1=0.0, L2=0.002")
plt.title('Predicted vs Actual Prices (L1=0.0, L2=0.002)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_wo_reg, scaler, "No Regularization")
plot_predictions(y_val, val_predictions_l1_l2_01, scaler, "L1=0.001, L2=0.001")
plot_predictions(y_val, val_predictions_l1_02, scaler, "L1=0.002, L2=0.0")
plot_predictions(y_val, val_predictions_l2_02, scaler, "L1=0.0, L2=0.002")
plt.title('Combined: Predicted vs Actual Prices')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    if len(X_test) > 0:
        test_predictions = model(X_test.to(device))
        plot_predictions(y_test, test_predictions, scaler, "Test")
        plt.title('Predicted vs Actual Prices (Test)')
        plt.legend()
        plt.show()

    last_40_days = X_test[-1].unsqueeze(0).to(device)