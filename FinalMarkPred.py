import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv("marks.csv")  # Replace "marks.csv" with the path to your dataset

# Preprocess the data
X = data.drop(columns=['Name', 'ID', 'Final Mark'])  # Dropping 'Name', 'ID', and 'Final Mark'
y = data['Final Mark']  # Target variable

# Train a machine learning model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_rmse = mean_squared_error(y_train, model.predict(X_train), squared=False)
test_rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Function to get user input
def get_user_input():
    while True:
        try:
            name = input("Enter student's name: ")
            student_id = int(input("Enter student's ID: "))
            quarterly_mark = float(input("Enter student's quarterly mark (0-100): "))
            if quarterly_mark < 0 or quarterly_mark > 100:
                raise ValueError("Quarterly mark should be between 0 and 100")
            halfyearly_mark = float(input("Enter student's half-yearly mark (0-100): "))
            if halfyearly_mark < 0 or halfyearly_mark > 100:
                raise ValueError("Half-yearly mark should be between 0 and 100")
            return name, student_id, quarterly_mark, halfyearly_mark
        except ValueError as e:
            print("Invalid input. Please try again.", e)

# Get input from the user
name, student_id, quarterly_mark, halfyearly_mark = get_user_input()

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'Quarterly Mark': [quarterly_mark],
    'Halfyearly Mark': [halfyearly_mark]
})

# Make prediction
final_mark_prediction = model.predict(input_data)

# Calculate confidence interval
# Assuming normal distribution of predictions
predictions = []
for _ in range(1000):  # Generate 1000 predictions
    predictions.append(model.predict(input_data)[0])
confidence_interval = np.percentile(predictions, [2.5, 97.5])

# Print prediction and confidence interval
print(f"Predicted final mark for {name} (ID: {student_id}): {final_mark_prediction[0]:.2f}")

