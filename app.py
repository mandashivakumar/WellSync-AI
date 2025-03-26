import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
file_path = "gym_members_exercise_tracking.csv"
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

# Normalize the features
scaler = StandardScaler()
X = data[features].values
X_scaled = scaler.fit_transform(X)

# Define the Multi-Task Neural Network
class MultiTaskMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiTaskMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        
        # Output layers for multi-task learning
        self.workout_output = nn.Linear(32, num_classes)  # Classification (Workout_Type)
        self.calories_output = nn.Linear(32, 1)  # Regression (Calories_Burned)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        
        workout = self.workout_output(x)  # Classification output
        calories = self.calories_output(x)  # Regression output
        return workout, calories

# Load the trained model
model = MultiTaskMLP(input_size=len(features), num_classes=len(np.unique(data["Workout_Type"])))
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Streamlit App
st.title("Gym Member Exercise Tracking")

# User Inputs
st.sidebar.header("Enter Your Details")
user_input = {}
for feature in features:
    if feature in categorical_features:
        options = label_encoders[feature].classes_
        user_input[feature] = st.sidebar.selectbox(f"Select {feature}", options)
    else:
        user_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

# Predict Button
if st.sidebar.button("Predict"):
    # Preprocess user input
    input_data = []
    for feature in features:
        if feature in categorical_features:
            value = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            value = user_input[feature]
        input_data.append(value)
    
    # Scale the input
    input_scaled = scaler.transform([input_data])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Get predictions
    with torch.no_grad():
        workout_pred, calories_pred = model(input_tensor)

    # Decode the predictions
    workout_type = torch.argmax(workout_pred, dim=1).item()
    calories_burned = calories_pred.item()

    # Inverse transform for calories
    calories_burned = (calories_burned * scaler.scale_[features.index("Calories_Burned")]) + scaler.mean_[features.index("Calories_Burned")]

    # Display predictions
    st.subheader("Predictions")
    st.write(f"Predicted Workout Type: {label_encoders['Workout_Type'].inverse_transform([workout_type])[0]}")
    st.write(f"Predicted Calories Burned: {calories_burned:.2f}")

    # Recommend workout
    recommendations = {
        "Yoga": ["Sun Salutation", "Warrior Pose", "Tree Pose"],
        "HIIT": ["Burpees", "Mountain Climbers", "Jump Squats"],
        "Strength": ["Deadlifts", "Bench Press", "Squats"],
        "Cardio": ["Running", "Cycling", "Jump Rope"]
    }
    workout_name = label_encoders["Workout_Type"].inverse_transform([workout_type])[0]
    st.subheader("Recommended Exercises")
    for exercise in recommendations.get(workout_name, ["General Fitness Routine"]):
        st.write(f"- {exercise}")

    # Provide nutrition guidance
    weight = user_input["Weight_(kg)"]
    height = user_input["Height_(m)"]
    age = user_input["Age"]
    bmr = 10 * weight + 6.25 * height * 100 - 5 * age + 5  # Basal Metabolic Rate
    daily_calories = bmr + calories_burned
    st.subheader("Personalized Nutrition Guidance")
    st.write(f"Your estimated daily calorie intake should be around {daily_calories:.2f} calories.")
