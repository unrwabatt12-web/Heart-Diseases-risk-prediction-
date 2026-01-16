# Heart Disease Risk Prediction System
## Project Overview

The Heart Disease Prediction System is a full-stack machine learning web application designed to predict the presence and severity of heart disease based on patient clinical data. The system integrates a trained machine learning model with a Flask REST API and an interactive frontend for real-time predictions.

This project is developed as part of an academic Machine Learning / AI deployment assignment and demonstrates end-to-end ML deployment, including model loading, API design, frontend integration, and robust error handling.


## Key Features

* Machine Learning Model for heart disease classification
* Flask REST API for predictions
* Interactive Frontend UI with auto-fill example data
* Probability Visualization with animated bars
* Color-coded Risk Levels based on severity
* Robust error handling & logging
* CORS-enabled for frontend-backend integration

## System Architecture

Frontend (HTML/CSS/JS)
        ↓  (JSON)
Flask REST API
        ↓
Machine Learning Model (.pkl)

## Project Structure

ITLML_801_S_A_25RP21655/
│
├── app_25RP21655.py          # Flask backend application
├── deployment/
│   ├── heart_disease_best_model.pkl
│   ├── feature_columns.txt
│   └── class_names.txt
│
├── templates/
│   └── index_25RP21655.html  # Frontend UI
│
├── README.md               
└── requirements.txt          # Python dependencies


## Dataset
- 5000 patient records
- 13 clinical and demographic features
- 5 diagnosis classes

## Models Trained
- MLP (ANN)
- Random Forest
- SVM
- KNN
- Gradient Boosting

## Example Features Used

The model expects the following input features:

* age
* sex
* cp (chest pain type)
* trestbps (resting blood pressure)
* chol (serum cholesterol)
* fbs (fasting blood sugar)
* restecg (resting ECG)
* thalach (maximum heart rate)
* exang (exercise induced angina)
* oldpeak (ST depression)
* slope
* ca (number of major vessels)
* thal

## Prediction Severity Colors

| Severity Level | Color       |
| -------------- | ----------- |
| No Disease     | Green       |
| Mild           | Light Green |
| Moderate       | Orange      |
| Severe         | Red         |
| Critical       | Dark Red    |

### Installation & Setup

### Clone the Repository

git clone <repository-url>
cd ITLML_801_S_A_25RP21655

### Install Dependencies

pip install -r requirements.txt

### Run the Application

python app_25RP21655.py

The app will be available at:
http://127.0.0.1:5000

## Technologies Used
Python
Flask (REST API)
Scikit-learn (ML model)
Pandas & NumPy
HTML / CSS / JavaScript
Bootstrap & Font Awesome

## Model Details

* Model type: Supervised classification model
* Trained on heart disease clinical dataset
* Serialized using `pickle / joblib`
* Supports probability predictions

## Error Handling

* Handles missing input features
* Graceful model load fallback strategies
* Frontend-safe response validation
* Logging for debugging and monitoring

## Learning Outcomes

* End-to-end ML deployment
* Backend–frontend integration
* Real-world error handling
* Model serialization and inference
* REST API design

## Author

Student ID:25RP21655
Course: ITLML 801 – Machine Learning Deployment
Institution: RP/ HUYE COLLEGE
