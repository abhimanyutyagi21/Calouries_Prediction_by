import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
MODEL_PATH = 'calories_predictor_model.joblib'
TEST_DATA_PATH = 'Newdata.csv'
TARGET_COLUMN = 'Calories_Burned'

def load_and_test_model():
    """Loads the saved model and pipeline, and makes a test prediction."""
    print(f"1. Loading the trained model from {MODEL_PATH}...")
    try:
        # Load the full pipeline (preprocessor and regressor)
        full_pipeline_loaded = joblib.load(MODEL_PATH)
        print("   -> Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file {MODEL_PATH} not found. Please run app.py first.")
        return

    print(f"2. Loading test data from {TEST_DATA_PATH}...")
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {TEST_DATA_PATH} not found. Using a synthetic test sample instead.")
        # If Newdata.csv isn't found, we use a single synthetic row for demonstration
        # This row MUST have the same columns as the training input X
        test_df = pd.DataFrame([{
            'Age': 35.0, 'Gender': 'Male', 'Weight (kg)': 70.0, 'Height (m)': 1.75, 
            'Max_BPM': 180.0, 'Avg_BPM': 150.0, 'Resting_BPM': 65.0, 
            'Session_Duration (hours)': 1.5, 'Fat_Percentage': 20.0, 
            'Water_Intake (liters)': 3.0, 'Workout_Frequency (days/week)': 4.0, 
            'Experience_Level': 2.0, 'BMI': 22.86, 'Daily meals frequency': 3.0, 
            'Physical exercise': 1.0, 'meal_type': 'Lunch', 
            'diet_type': 'Standard', 'sugar_g': 20.0, 'sodium_mg': 1500.0, 
            'cholesterol_mg': 100.0, 'serving_size_g': 250.0, 
            'cooking_method': 'Boiled', 'prep_time_min': 10.0, 
            'cook_time_min': 20.0, 'rating': 4.5, 'is_healthy': 1.0, 
            'Difficulty Level': 'Beginner', 'Body Part': 'Core', 
            'Type of Muscle': 'Slow twitch', 'Workout': 'Planks',
            # Include dummy values for the columns that were dropped in training
            'Name of Exercise': 'Squats', 'Sets': 3, 'Reps': 10, 'Benefit': 'Stronger Legs',
            'Burns Calories (per 30 min)': 150.0, 'Target Muscle Group': 'Quads', 
            'Equipment Needed': 'None', 'meal_name': 'Test Meal',
            'Calories': 2000, 'Carbs': 200, 'Proteins': 100, 'Fats': 50,
            TARGET_COLUMN: 1000 # Dummy target value
        }])


    # --- Extract Features and Target ---
    # NOTE: The feature extraction MUST match app.py's DROPPED_COLUMNS logic
    
    # These columns were dropped in app.py before training X
    DROP_COLUMNS = [
        # Redundant/Derived columns
        'Name of Exercise', 'Sets', 'Reps', 'Benefit', 
        'Burns Calories (per 30 min)', 'Target Muscle Group', 'Equipment Needed',
        'Type of Muscle', 'Workout', 'meal_name', 'is_healthy',
        'BMI_calc', 'cal_from_macros', 'Burns Calories (per 30 min)_bc', 
        'cal_balance', 'lean_mass_kg', 'expected_burn', 
        # Target variable and highly correlated derived macros
        'Calories', 'Carbs', 'Proteins', 'Fats', TARGET_COLUMN
    ]

    # Drop columns not used for prediction from the test set
    features_to_keep = [col for col in test_df.columns if col not in DROP_COLUMNS]
    X_test_sample = test_df[features_to_keep]

    # If the file had a target column, we can calculate metrics. Otherwise, just predict.
    y_true = None
    if TARGET_COLUMN in test_df.columns:
        y_true = test_df[TARGET_COLUMN].iloc[0]

    # --- Prediction ---
    print("\n3. Making prediction on the test sample...")
    
    # The loaded pipeline automatically applies Box-Cox, Scaling, and Encoding 
    # based on the training data statistics.
    predicted_calories = full_pipeline_loaded.predict(X_test_sample.iloc[[0]])[0]
    
    print("------------------------------------------")
    print("   Test Sample Data Point:")
    print(f"   Gender: {X_test_sample['Gender'].iloc[0]}, Weight: {X_test_sample['Weight (kg)'].iloc[0]} kg")
    print(f"   Workout Type: {X_test_sample['Workout_Type'].iloc[0]}, Duration: {X_test_sample['Session_Duration (hours)'].iloc[0]} hours")
    print("------------------------------------------")
    print(f"   🔥 Predicted Calories Burned: {predicted_calories:.2f}")
    if y_true is not None:
         print(f"   Actual Calories Burned: {y_true:.2f}")
    print("------------------------------------------")


if __name__ == "__main__":
    load_and_test_model()