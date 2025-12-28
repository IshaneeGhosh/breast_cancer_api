Breast Cancer Tumor Classification API

A machine learning–powered REST API that predicts whether a breast tumor is Benign or Malignant using the Wisconsin Breast Cancer dataset.
The model is trained using Logistic Regression and deployed using FastAPI with live, interactive Swagger documentation.

Project Overview
This project demonstrates an end-to-end machine learning workflow, including:

Data cleaning and preprocessing
Feature scaling and model training
Model evaluation using classification metrics
Deployment as a RESTful API
Real-time inference via Swagger UI
The goal is to showcase how a trained ML model can be served reliably in a production-like environment.

Machine Learning Details
Dataset: Wisconsin Diagnostic Breast Cancer Dataset
Task: Binary classification
Classes:
0 → Benign
1 → Malignant
Algorithm: Logistic Regression
Number of Features: 30 (mean, standard error, and worst measurements)

Tech Stack:-

Data & ML-
Python
Pandas – data loading & preprocessing
NumPy – numerical operations
Seaborn – data visualization
Scikit-learn
StandardScaler
LogisticRegression
train_test_split
Evaluation metrics

Backend & Deployment-
FastAPI – REST API framework
Pydantic – input validation
Pickle – model serialization
Swagger UI – API testing & documentation
Render – cloud deployment

Project Structure
breast_cancer_api/
│
├── app.py                 # FastAPI application
├── data.csv               # Dataset
├── logistic_model.pkl     # Trained ML model
├── scaler.pkl             # Trained scaler
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

Data Preprocessing:-

Removed irrelevant columns (id, Unnamed: 32)
Converted target labels:
M → 1 (Malignant)
B → 0 (Benign)
Applied StandardScaler for feature normalization
Split data into training (70%) and testing (30%)

Model Evaluation:-
The model achieved high accuracy (~95%+) on the test dataset.
Evaluation metrics include:
Accuracy
Precision
Recall
F1-score
These metrics ensure reliable performance, especially for medical classification tasks where false negatives are critical.

API Endpoints:-
Health Check
GET /

Response

{
  "status": "Breast Cancer API is running"
}


Tumor Prediction
POST /predict


Request Body

{
  "features": [30 numerical feature values]
}

for example-
{
  "features": [
    20.5, 26.3, 135.1, 1290.0, 0.109, 0.183, 0.227, 0.125, 0.216, 0.072,
    0.755, 2.463, 5.203, 128.0, 0.012, 0.055, 0.064, 0.031, 0.036, 0.009,
    25.3, 34.5, 168.2, 2019.0, 0.162, 0.468, 0.560, 0.280, 0.404, 0.124
  ]
}


Response

{
  "prediction": "Benign"
}


or

{
  "prediction": "Malignant"
}

The API requires exactly 30 feature values, matching the training dataset.

Interactive API Testing
Once deployed, the API provides interactive Swagger documentation:
https://breast-cancer-api-1-8doi.onrender.com/docs


We can:
Test predictions directly in the browser
Validate inputs automatically
View request/response schemas

Key Learnings:-
Importance of feature scaling in ML pipelines
Handling real-world deployment errors (500 vs 422 errors)
Designing ML-backed APIs with validation
Serving models efficiently without retraining
Debugging input shape and feature mismatches

⚠️ Disclaimer:-
This project is intended for educational and demonstration purposes only.
It is not a medical diagnostic tool and should not be used for clinical decision-making.
