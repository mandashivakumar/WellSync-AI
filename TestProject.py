import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
file_path = "C:/Users/PCLab/Downloads/gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

# Data Preprocessing
# Encoding categorical variables
label_encoders = {}
categorical_features = ["Gender", "Experience_Level", "Workout_Type"]

for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Handling missing values by dropping them
data = data.dropna()

# Selecting feature columns
features = [
    "Age", "Gender", "Weight_(kg)", "Height_(m)", "Max_BPM", "Avg_BPM",
    "Resting_BPM", "Session_Duration (hours)", "Calories_Burned",
    "Fat_Percentage", "Water_Intake (liters)", "Workout_Frequency", "BMI"
]

# Splitting data into training and testing sets
X = data[features].values
y = data["Workout_Type"].values  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the Improved Neural Network
class MLPWorkoutModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPWorkoutModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Increased neurons
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        return x

# Get number of unique classes in the target variable
num_classes = len(np.unique(y_train))

# Initialize the model
model = MLPWorkoutModel(input_size=len(features), num_classes=num_classes)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 Regularization
loss_fn = nn.CrossEntropyLoss()

# Training Loop with Accuracy Tracking
epochs = 50
batch_size = 32
train_loader = torch.utils.data.DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    train_accuracy = 100 * correct / total
    if (epoch + 1) % 5 == 0:  # Print every 5 epochs
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

# Model Evaluation on Test Set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_predictions = torch.max(test_outputs, 1)
    test_accuracy = 100 * (test_predictions == y_test_tensor).sum().item() / y_test_tensor.size(0)

print(f"\nFinal Model Accuracy on Test Data: {test_accuracy:.2f}%")


# User Input Prediction
def predict_fitness():
    user_input = []
    for feature in features:
        value = float(input(f"Enter {feature}: "))
        user_input.append(value)

    user_input_scaled = scaler.transform([user_input])
    user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

    # Get prediction
    prediction = model(user_input_tensor).detach().numpy()
    workout_type_pred = np.round(prediction[0, 0])
    calories_pred = prediction[0, 1]  # Raw model output for calories

    # 🔥 Rescale Calories Back to Original Scale
    calories_pred = (calories_pred * scaler.scale_[features.index("Calories_Burned")]) + scaler.mean_[
        features.index("Calories_Burned")]

    # Output predictions
    print(f"Predicted Workout Type: {label_encoders['Workout_Type'].inverse_transform([int(workout_type_pred)])[0]}")
    print(f"Predicted Calories Burned: {calories_pred:.2f}")


predict_fitness()
