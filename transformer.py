import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv('filtered_data.csv')  # Replace 'your_file.csv' with your actual file path

# Select only the 'Date' and 'Point' columns
df_cleaned = df[['Date', 'Point']]

# Convert 'Date' column to datetime format
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])

# Feature engineering on Date column for discharge forecasting
df_cleaned['Year'] = df_cleaned['Date'].dt.year
df_cleaned['Month'] = df_cleaned['Date'].dt.month
df_cleaned['Day'] = df_cleaned['Date'].dt.day
df_cleaned['DayOfWeek'] = df_cleaned['Date'].dt.dayofweek
df_cleaned['IsWeekend'] = df_cleaned['DayOfWeek'].isin([5, 6]).astype(int)
df_cleaned['DayOfYear'] = df_cleaned['Date'].dt.dayofyear

# Adding lag features for time-series prediction
df_cleaned['Point_Lag1'] = df_cleaned['Point'].shift(1)
df_cleaned['Point_Lag7'] = df_cleaned['Point'].shift(7)
df_cleaned['Point_Lag30'] = df_cleaned['Point'].shift(30)

# Adding rolling statistics
df_cleaned['RollingMean_7'] = df_cleaned['Point'].rolling(window=7).mean()
df_cleaned['RollingMean_30'] = df_cleaned['Point'].rolling(window=30).mean()

# Drop rows with NaN values created by lag features
df_cleaned = df_cleaned.dropna()

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_cleaned.drop(columns=['Date']))

# Split data into training and test sets
train_data, test_data = train_test_split(df_scaled, test_size=0.2, shuffle=False)

# Create dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length, :-1]
        y = self.data[index + self.seq_length, -1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Create data loaders
seq_length = 30
train_dataset = TimeSeriesDataset(train_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

# Initialize model
input_size = train_data.shape[1] - 1
model = TransformerModel(input_size=input_size, d_model=64, nhead=4, num_layers=2, output_size=1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

# Evaluation loop
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
    print(f'Test Loss: {total_loss/len(test_loader):.4f}')

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, epochs=20)
evaluate_model(model, test_loader, criterion)

