import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Load data
data = pd.read_csv("/Users/colepuls/CS/MUHackathon2025/Raw_data/diabetes.csv")

# Count duplicate how many duplicate rows exist
num_dups = data.duplicated().sum()

# Remove any dups
if num_dups > 0:
    data.drop_duplicates(inpalce=True)

# List columns where zeros are likely invalid or indicate missing data
cols_to_clean = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Replace 0 with nan, indicates missing value
for c in cols_to_clean:
    data[c].replace(0, np.nan, inplace=True)

# Check how many nans are in columns
data[cols_to_clean].isnull().sum()

# Impute missing values, fill with column means
data[cols_to_clean] = data[cols_to_clean].fillna(data[cols_to_clean].mean())

# Split targets
x = data.drop(columns=["Outcome"]).values
y = data["Outcome"].values

# Convert into tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long) # for classification labels

# Calculate mean and std for each feature (column)
mean = x_tensor.mean(dim=0)
std = x_tensor.std(dim=0)

std[std == 0] = 1e-7 # avoid division by zero

# Normalize!!!
x_tensor = (x_tensor - mean) / std # x tensor is now normalized column wise

class DiabetesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# initialize dataset    
dataset = DiabetesDataset(x_tensor, y_tensor)

sample_features = dataset[0]
sample_labels = dataset[0]

# Split into training and validation sets

# 80/20 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# test
#for batch_features, batch_labels in train_loader:
#    print(batch_features.shape, batch_labels.shape)
#    break

# compute stats after normalization
check_mean = torch.mean(torch.vstack([dataset[i][0] for i in range(len(dataset))]), dim=0)
check_std = torch.std(torch.vstack([dataset[i][0] for i in range(len(dataset))]), dim=0)

# neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = SimpleNet(input_dim=x.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
    
# Training
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

# Validation
model.eval()
val_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for val_features, val_labels in val_loader:
        val_outputs = model(val_features)
        loss = criterion(val_outputs, val_labels)
        val_loss += loss.item()

        _, predicted = torch.max(val_outputs, dim=1)
        correct += (predicted == val_labels).sum().item()
        total += val_labels.size(0)
avg_val_loss = val_loss / len(val_loader)
accuracy = 100 * correct / total

print(f"Epoch {epoch+1}/{epochs} | "
      f"Train Loss: {avg_train_loss:.4f} | "
      f"Val Loss: {avg_val_loss:.4f} | "
      f"Val Acc: {accuracy:.2f}%")

# Walkthrough
print("Please enter the patient's details:")
p = float(input("Pregnancies: ")) # number of pregnancies
gl = float(input("Glucose: ")) # glucose level
bp = float(input("BloodPressure: "))
st = float(input("SkinThickness: "))
ins = float(input("Insulin: ")) # insulin level
bmi = float(input("BMI: ")) # weight relative to height
dpf = float(input("DiabetesPedigreeFunction: ")) # function that estimates likelyness of having diabetes based off family history
age = float(input("Age: "))

new_patient = [p, gl, bp, st, ins, bmi, dpf, age]

# Convert to tensor and normalize
patient_tensor = torch.tensor(new_patient, dtype=torch.float32)
patient_tensor = (patient_tensor - mean) / std
patient_tensor = patient_tensor.unsqueeze(0)  # shape: (1, 8)

# Predict
model.eval()
with torch.no_grad():
    output = model(patient_tensor)
    _, predicted_class = torch.max(output, dim=1)
    predicted_class = predicted_class.item()

if predicted_class == 1:
    print("\nThis patient is likely to have diabetes.")
else:
    print("\nThis patient is not likely to have diabetes.")

probs = F.softmax(output, dim=1)
p_diabetes = probs[0, 1].item()
p_diabetesPercentage = p_diabetes * 100
print(f"Probability of diabetes: {p_diabetesPercentage:.2f}%")
