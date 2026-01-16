import requests
from datetime import datetime

# LOAD DEPLOYMENT FILES

try:
    with open('deployment/feature_columns.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    with open('deployment/class_names.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    with open('deployment/heart_disease_best_model.pkl', 'rb') as f:
        model_data = f.read()
    
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    exit()

# TEST PATIENT DATA

test_patients = [
    {
        "name": "52-year-old Male with Typical Angina",
        "age": 52, "sex": 1, "cp": 0,
        "trestbps": 140, "chol": 230, "fbs": 0,
        "restecg": 0, "thalach": 160, "exang": 0,
        "oldpeak": 1.0, "slope": 0, "ca": 0, "thal": 0
    },
    {
        "name": "61-year-old Female with Atypical Angina",
        "age": 61, "sex": 0, "cp": 1,
        "trestbps": 150, "chol": 260, "fbs": 1,
        "restecg": 1, "thalach": 140, "exang": 1,
        "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 1
    }
]

# INITIALIZE COUNTERS

passed = 0
failed = 0

# PREDICTION TEST
print("HEART DISEASE PREDICTION - RESULTS")
print(f"Analysis Date: {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n")

for i, patient_data in enumerate(test_patients, 1):
    patient_name = patient_data.pop('name')
    
    print(f"TEST {i}: {patient_name}")
    print(f"Age: {patient_data['age']} | Sex: {'Male' if patient_data['sex']==1 else 'Female'}")
    print(f"BP: {patient_data['trestbps']} mmHg | Chol: {patient_data['chol']} mg/dl | HR: {patient_data['thalach']} bpm")
    
    try:
        response = requests.post('http://localhost:5000/predict', json=patient_data, timeout=5)
        result = response.json()
        
        if 'error' in result:
            print(f"ERROR: {result['error']}\n")
            failed += 1
        else:
            # Extract prediction data
            prediction = result.get('predicted_class_label') or result.get('prediction')
            confidence = result.get('confidence', 0)
            color = result.get('color', {} )
            probs = result.get('probabilities', {})
            
            # Display results
            print(f"\n PREDICTION RESULTS")
            print(f"{'┌' + '─' * 78 + '┐'}")
            print(f"│ Diagnosis: {prediction.upper():30s} Risk Level: {color.upper():20s} Confidence: {confidence:5.1f}% │")
            print(f"{'├' + '─' * 78 + '┤'}")
            
            # Probability bars
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            for cls, prob in sorted_probs:
                bar_length = int(prob / 5)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"│ {cls:20s} {bar} {prob:6.2f}% │")
            
            print(f"{'└' + '─' * 78 + '┘'}\n")
            passed += 1
    
    except requests.exceptions.ConnectionError:
        print(f"CONNECTION ERROR: Could not reach http://localhost:5000")
        print(f"Make sure Flask app is running: python app_25RP21655.py\n")
        failed += 1
    except requests.exceptions.Timeout:
        print(f"TIMEOUT ERROR: Server took too long to respond\n")
        failed += 1
    except Exception as e:
        print(f"ERROR: {str(e)}\n")
        failed += 1

# TEST SUMMARY
print("TEST SUMMARY")
print(f"Total Tests: {len(test_patients)}")
print(f"Analysis completed: {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")