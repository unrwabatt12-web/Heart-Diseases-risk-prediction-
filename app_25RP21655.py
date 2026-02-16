from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Setup logging
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route("/favicon.ico")
def favicon():
    # /vercel.svg is automatically served when included in the public/** directory.
    return redirect("/vercel.svg", code=307)

# Load model and metadata
MODEL_PATH = 'deployment/heart_disease_best_model.pkl'
FEATURES_PATH = 'deployment/feature_columns.txt'
CLASSES_PATH = 'deployment/class_names.txt'

def load_model():
    """Load the trained model with multiple fallback strategies."""
    model_loaders = [
        lambda: joblib.load(MODEL_PATH) if HAS_JOBLIB else None,
        lambda: pickle.load(open(MODEL_PATH, 'rb'), encoding='latin1'),
        lambda: pickle.load(open(MODEL_PATH, 'rb')),
    ]
    
    for loader in model_loaders:
        try:
            if loader:
                model = loader()
                logger.info("Model loaded successfully")
                return model
        except Exception as e:
            logger.warning(f"Model loading attempt failed: {e}")
            continue
    
    logger.error("All model loading attempts failed")
    return None

def load_features():
    """Load feature column names."""
    try:
        with open(FEATURES_PATH, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(features)} features")
        return features
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        # Default features based on typical heart disease dataset
        return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 
                'slope', 'ca', 'thal']

def load_classes():
    """Load class names."""
    try:
        with open(CLASSES_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(classes)} classes: {classes}")
        return classes
    except Exception as e:
        logger.error(f"Error loading classes: {e}")
        return ['No Heart Disease', 'Heart Disease Present']

# Load resources
model = load_model()
feature_columns = load_features()
class_names = load_classes()

# Validate model is ready
if model is None:
    logger.error("CRITICAL: Model failed to load. Please check the model file.")
else:
    logger.info("Model loaded successfully")
    logger.info(f"Features expected: {feature_columns}")
    logger.info(f"Classes: {class_names}")

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index_25RP21655.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'features_loaded': len(feature_columns) > 0,
        'classes_loaded': len(class_names) > 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Build input dictionary without forcing float
        input_data = {}
        missing_features = []

        for col in feature_columns:
            if col in data:
                input_data[col] = data[col]
            else:
                missing_features.append(col)

        if missing_features:
            return jsonify({
                'error': 'Missing features',
                'missing_features': missing_features
            }), 400

        # Create DataFrame
        input_df = pd.DataFrame([input_data])[feature_columns]

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        # Prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Handle prediction type
        if isinstance(prediction, str):
         pred_label = prediction
         pred_idx = class_names.index(prediction)
        else:
         pred_idx = int(prediction)
         pred_label = class_names[pred_idx]

        # Color coding
        color_map = {
              0: "green",
              1: "lightgreen",
              2: "orange",
              3: "red",
              4: "darkred"
             }

        prob_dict = {
            class_names[i]: round(float(prob * 100), 2)
        for i, prob in enumerate(probabilities)
           }

        result = {
            "predicted_class_index": pred_idx,
            "predicted_class_label": pred_label,
            "color": color_map[pred_idx],
            "confidence": round(max(probabilities) * 100, 2),
            "probabilities": prob_dict,
            "timestamp": datetime.now().isoformat()
        }


        logger.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/features', methods=['GET'])
def get_features():
    """Return the list of required features."""
    return jsonify({
        'features': feature_columns,
        'count': len(feature_columns)
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Return the list of class names."""
    return jsonify({
        'classes': class_names,
        'count': len(class_names)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
