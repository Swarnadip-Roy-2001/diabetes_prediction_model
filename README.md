# PIMA Diabetes Prediction System

This repository contains a Machine Learning workflow using a Support Vector Machine (SVM) classifier to predict diabetes onset based on diagnostic measurements.

## Project Structure

- `diabetes_model.sav`: The trained Support Vector Machine (SVM) model.
- `scaler.sav`: The fitted `StandardScaler` used to normalize incoming feature metrics.
- `requirements.txt`: The required dependencies (`numpy`, `pandas`, `scikit-learn`) to run the system.
- `.gitignore`: Specifying untracked and temporary local environment files.
- `README.md`: Project description and usage manual.

## Local Environment Setup

1. **Clone or Download** this directory to your local computer.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows:** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`
4. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to Make Predictions Locally

```python
import pickle
import numpy as np

# 1. Load the model and scaler
with open('diabetes_model.sav', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# 2. Prepare sample input data
# Example: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_array = np.asarray(input_data).reshape(1, -1)

# 3. Scale and Predict
scaled_data = scaler.transform(input_array)
prediction = classifier.predict(scaled_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
```
