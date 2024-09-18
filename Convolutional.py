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

X = X.reshape(-1, input_len, 1)

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

class CNN2D(nn.Module):
    def __init__(self, input_len):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        
        self.fc1 = nn.Linear(64 * input_len, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def loss_with_regularization(predictions, targets, model, l1_lambda=0.002, l2_lambda=0.002):
    mse_loss = nn.MSELoss()(predictions, targets)

    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    
    total_loss = mse_loss + l1_lambda * l1_norm + l2_lambda * l2_norm
    return total_loss

model = CNN2D(input_len)
criterion = nn.MSELoss()

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

def plot_predictions(y_val, predictions, scaler, label):
    y_val_inverse = scaler.inverse_transform(y_val.cpu().numpy())
    predictions_inverse = scaler.inverse_transform(predictions.cpu().numpy())

    plt.plot(y_val_inverse, label=f'Actual Prices ({label})', color='b')
    plt.plot(predictions_inverse, label=f'Predicted Prices ({label})', linestyle='dashed')

epochs = 100
batch_size = 32

cnn2d_model = CNN2D(input_len)
val_predictions_cnn2d = train_model(cnn2d_model, X_train, y_train, X_val, y_val, epochs, batch_size)
plt.figure(figsize=(10, 5))
plot_predictions(y_val, val_predictions_cnn2d, scaler, "2D CNN")
plt.title('Predicted vs Actual Prices (2D CNN)')
plt.legend()
plt.show()

#with torch.no_grad():
    #if len(X_test) > 0:
        #test_predictions = cnn2d_model(X_test)
        #plt.figure(figsize=(10, 5))
        #plot_predictions(y_test, test_predictions, scaler, "2D CNN Test")
        #plt.title('Test: Predicted vs Actual Prices (2D CNN)')
        #plt.legend()
        #plt.show()

with torch.no_grad():
    last_40_days = X_test[-1].unsqueeze(0)
    next_day_prediction = cnn2d_model(last_40_days)
    next_day_price = scaler.inverse_transform(next_day_prediction.numpy())
    print("Predicted price for the next day:", next_day_price[0][0])