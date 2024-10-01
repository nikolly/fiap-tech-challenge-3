from flask import Flask, request, Blueprint, jsonify
from src.core.train_model import train_machine_learning_model
from flask_cors import CORS
import pickle
import os
import pandas as pd
import logging


api_bp = Blueprint('api', __name__, url_prefix='/api')

# Load configuration from environment variables or config files
BUCKET_NAME = os.getenv('BUCKET_NAME', 'openweather-tc3')
S3_FOLDER = os.getenv('S3_FOLDER', 'Silver')
LOCAL_FOLDER = os.getenv('LOCAL_FOLDER', 'data')
MODEL_PATH = os.getenv('MODEL_PATH', 'modelo/modelo_regressao_linear.pkl')

# Load the model at startup
model = None


@api_bp.route('/train', methods=['POST'])
def train():
    """
    Endpoint to train the machine learning model.
    
    Returns:
        JSON response indicating success or error.
    """
    global model
    response, status = train_machine_learning_model(BUCKET_NAME, S3_FOLDER, LOCAL_FOLDER, MODEL_PATH)
    if status == 200:
        # Reload the model if training was successful
        from src.core.load_model import load_model
        model = load_model(MODEL_PATH)
    return jsonify(response), status


@api_bp.route('/prediction', methods=['POST'])
def create_prediction():
    """
    Endpoint to generate a prediction from the trained model.
    
    Expects:
        JSON payload with 'temp_max' and 'temp_afternoon' fields.
    
    Returns:
        JSON response with predicted humidity or an error message.
    """
    if model is None:
        return jsonify({'error': 'Model not trained'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate input data
        required_fields = ['temp_max', 'temp_afternoon']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing field '{field}'"}), 400
            if not isinstance(data[field], (int, float)):
                return jsonify({'error': f"Invalid type for field '{field}'"}), 400

        # Convert input data to DataFrame for prediction
        input_data = pd.DataFrame([data])

        # Make prediction
        humidity = model.predict(input_data)
        return jsonify({'humidity': round(humidity[0], 1)})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500
