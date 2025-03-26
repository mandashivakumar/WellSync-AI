import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 28px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        width: 50%;
        margin: 0 auto;
        display: block;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 5px;
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 24px;
        text-align: center;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-bottom: 16px;
    }
    .small-subheader {
        style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px;"
    }
    .stMarkdown {
        margin-bottom: 16px;
    }
    .stSuccess {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 16px;
    }
    .stError {
        font-size: 16px;
        color: #ff4b4b;
        margin-bottom: 16px;
    }
    .section {
        margin-bottom: 32px;
    }
    .input-group-border {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        background-color: #f9f9f9;
    }
    .input-group {
        display: flex;
        gap: 16px;
        margin-bottom: 16px;
    }
    .input-group > div {
        flex: 1;
    }
    .recommendations {
        background-color: #f9f9f9;
        padding: 6px;
        border-radius: 8px;
        margin-bottom: 6px;
    }
    .recommendations ul {
        margin: 0;
        padding-left: 20px;
    }
    .recommendations li {
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Selecting feature columns
features = [
    "Age", "Gender", "Weight_(kg)", "Height_(m)", "Max_BPM", "Avg_BPM",
    "Resting_BPM", "Session_Duration (hours)", "Calories_Burned",
    "Fat_Percentage", "Water_Intake (liters)", "Workout_Frequency", "BMI"
]

# Load the trained model
model = MultiTaskMLP(input_size=len(features), num_classes=len(np.unique(data["Workout_Type"])))
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load the scaler
scaler = StandardScaler()
scaler.fit(data[features].values)

# Login Page
def login_page():
    st.markdown('<div class="title">Login Page</div>', unsafe_allow_html=True)
    email = st.text_input("Enter your email ID")
    if st.button("Login"):
        if email:  # Check if email is not empty
            st.session_state.logged_in = True
            st.session_state.email = email
            st.rerun()
        else:
            st.error("Please enter a valid email ID")

# Recommend Workout
def recommend_workout(workout_type):
    recommendations = {
        "Yoga": ["Sun Salutation", "Warrior Pose", "Tree Pose"],
        "HIIT": ["Burpees", "Mountain Climbers", "Jump Squats"],
        "Strength": ["Deadlifts", "Bench Press", "Squats"],
        "Cardio": ["Running", "Cycling", "Jump Rope"]
    }
    workout_name = label_encoders["Workout_Type"].inverse_transform([workout_type])[0]
    
    st.write("")
    # Display Recommended Exercises with green background
    st.markdown(
        f'<div class="small-subheader" style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px;">'
        f'Recommended Exercises'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # List exercises
    with st.container():
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown('<ul>', unsafe_allow_html=True)
        for exercise in recommendations.get(workout_name, ["General Fitness Routine"]):
            st.markdown(f'<li>{exercise}</li>', unsafe_allow_html=True)
        st.markdown('</ul></div>', unsafe_allow_html=True)

# Provide Nutrition Guidance
def provide_nutrition_guidance(calories_burned, weight, height, age):
    bmr = 10 * weight + 6.25 * height * 100 - 5 * age + 5  # Basal Metabolic Rate
    daily_calories = bmr + calories_burned
    
    st.write("")
    # Display Personalized Nutrition Guidance with green background
    st.markdown(
        f'<div class="small-subheader" style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px;">'
        f'Personalized Nutrition Guidance'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Display daily calorie intake
    st.write(f"Your estimated daily calorie intake should be around {daily_calories:.2f} calories.")

# Main App
def main_app():
    st.markdown('<div class="title">Gym Member Exercise Tracking</div>', unsafe_allow_html=True)
    st.write(f"Welcome, {st.session_state.email}!")

    # Add a guiding sentence
    st.markdown("Please enter your details below to get personalized workout and nutrition recommendations.", unsafe_allow_html=True)

    # Wrap input fields in a bordered container
    #st.markdown('<div class="input-group-border">', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown('<div class="subheader">Your Details</div>', unsafe_allow_html=True)

    # Inputs in two columns
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", min_value=0.0, value=75.0)
        height = st.number_input("Height (m)", min_value=0.0, value=1.75)
        max_bpm = st.number_input("Max BPM", min_value=0, value=180)
        avg_bpm = st.number_input("Avg BPM", min_value=0, value=150)
        resting_bpm = st.number_input("Resting BPM", min_value=0, value=70)

    with col2:
        session_duration = st.number_input("Session Duration (hours)", min_value=0.0, value=1.5)
        calories_burned = st.number_input("Calories Burned", min_value=0, value=500)
        fat_percentage = st.number_input("Fat Percentage", min_value=0.0, value=20.0)
        water_intake = st.number_input("Water Intake (liters)", min_value=0.0, value=2.5)
        workout_frequency = st.number_input("Workout Frequency", min_value=0, value=4)
        bmi = st.number_input("BMI", min_value=0.0, value=24.0)

    st.markdown('</div>', unsafe_allow_html=True)  # Close the bordered container

    # Predict Button
    if st.button("Predict"):
        # Preprocess user input
        user_input = [
            age,
            label_encoders["Gender"].transform([gender])[0],
            weight,
            height,
            max_bpm,
            avg_bpm,
            resting_bpm,
            session_duration,
            calories_burned,
            fat_percentage,
            water_intake,
            workout_frequency,
            bmi
        ]

        # Scale the input
        user_input_scaled = scaler.transform([user_input])
        user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

        # Get predictions
        model.eval()
        with torch.no_grad():
            workout_pred, calories_pred = model(user_input_tensor)

        # Decode the predictions
        workout_type = torch.argmax(workout_pred, dim=1).item()
        calories_burned_pred = calories_pred.item()

        # Inverse transform for calories
        calories_burned_pred = (calories_burned_pred * scaler.scale_[features.index("Calories_Burned")]) + scaler.mean_[features.index("Calories_Burned")]

        # Display predictions in a table
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Results</div>', unsafe_allow_html=True)  # Larger font size

       # Display predicted workout type
        st.markdown(
            f'<div class="small-subheader" style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px;">'
            f'Predicted Workout Type: {label_encoders["Workout_Type"].inverse_transform([workout_type])[0]}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Display predicted calories burned
        st.markdown(
            f'<div class="small-subheader" style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px;">'
            f'Predicted Calories Burned: {calories_burned_pred:.2f}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Recommend workout
        recommend_workout(workout_type)

        # Provide nutrition guidance
        provide_nutrition_guidance(calories_burned_pred, weight, height, age)
        st.markdown('</div>', unsafe_allow_html=True)

# Check if user is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()