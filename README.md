# CVDs-Detection
# Overview
Primary Goal: Develop a robust machine learning model to accurately predict the risk of heart failure in patients.

Data Utilization: Leverage comprehensive clinical data, including patient history and key health indicators, for model development.

Objective: Facilitate early detection and timely intervention to manage heart failure risk.

Outcome: Significantly improve patient outcomes and reduce mortality rates associated with cardiovascular diseases.

Motivation
Global Impact: Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible for 17.9 million deaths annually, with heart attacks and strokes causing most of these deaths.

Predictive Model Development: Motivated by the need to develop a predictive model for heart failure using a dataset with 11 key features to aid early detection and management for high-risk individuals.

# Challenges:

Ensuring data accuracy.

Selecting relevant features.

Creating a model that generalizes well across diverse populations.

Integrating the model into clinical practice.

Driving Force: The potential to save lives drives the efforts despite these challenges.

# Novelty
Accessible Health Intelligence: The system leverages machine learning insights to assist healthcare providers and individuals in making informed decisions, promoting early intervention and effective management of CVD risks.

Health Suggestions Based on Predictions: Using machine learning predictions to provide personalized health suggestions for individuals. These recommendations aim to help users mitigate risks associated with cardiovascular diseases by addressing factors such as lifestyle, diet, and physical activity.

Predictive Modeling for CVDs: Implementation of hybrid machine learning models combining multiple algorithms (e.g., Random Forest, Logistic Regression, SVC, KNN, Decision Tree, and XGBoost) to enhance the accuracy and robustness of cardiovascular disease (CVD) prediction.

# Overview of the Methodology
##  Data Preprocessing:

Cleaned the dataset by removing duplicates and outliers.

Created a new feature, BMI (Body Mass Index), for better predictions.

Split the data into training and testing sets (80:20 ratio).

## Model Development:

Trained six machine learning models:

Random Forest

Logistic Regression

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Decision Tree

Gradient Boosting

Saved the trained models using the Joblib library for efficient reuse.

## Backend Implementation:

Built a Flask-based backend to handle predictions.

The backend receives input data, processes it through the models, averages the predictions, and sends the final result to the frontend.
