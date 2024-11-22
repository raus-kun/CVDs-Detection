# Heart Diseases Prediction Model
## Overview
1) Primary Goal: Develop a robust machine learning model to accurately predict the risk of heart failure in patients.

2) Data Utilization: Leverage comprehensive clinical data, including patient history and key health indicators, for model development.

3) Objective: Facilitate early detection and timely intervention to manage heart failure risk.

4) Outcome: Significantly improve patient outcomes and reduce mortality rates associated with cardiovascular diseases.


## Overview of the Methodology
### 1) Data Preprocessing:

Cleaned the dataset by removing duplicates and outliers.

Created a new feature, BMI (Body Mass Index), for better predictions.

Split the data into training and testing sets (80:20 ratio).

### 2) Model Development:
Trained six machine learning models:

1) Random Forest

2) Logistic Regression

3) Support Vector Classifier (SVC)

4) K-Nearest Neighbors (KNN)

5) Decision Tree

6) Gradient Boosting

Saved the trained models using the Joblib library for efficient reuse.

### 3) Backend Implementation:

Built a Flask-based backend to handle predictions.

The backend receives input data, processes it through the models, averages the predictions, and sends the final result to the frontend.

## Design Workflow
### 1) Input Data:

User enters feature values (age, gender, height, etc.) via the frontend.

### 2) Backend Processing:

Input is sent to the backend via an API request.

Features are passed to each ML model for prediction.

The average of model outputs determines the final prediction.

### 3) Output:

Final prediction (e.g., "High Risk of Heart Failure") is sent back to the frontend for display.

## Steps to Run the Project
### 1) Install Required Libraries
pip install numpy pandas matplotlib flask django requests scikit-learn tensorflow keras joblib
Download the Pre-trained Model File

### 2) Download the .pkl file from the following link:
https://drive.google.com/drive/folders/1doG0jk21IvYo7TzDsvhHXHiSziKPH8b8

Save the file in your project directory.

### 3) Run the Application

Execute the app.py file.
Wait until the program initializes and starts running.

### 4) Access the Application

Open your web browser and navigate to: http://127.0.0.1:5000
