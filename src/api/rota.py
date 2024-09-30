from flask import Flask, Blueprint, jsonify
from src.core.train_model import train_machine_learning_model
from src.core.predict import predict


app = Flask(__name__)
api_bp = Blueprint('api', __name__, url_prefix='/api')


# POST /api/data
@api_bp.route('/train', methods=['POST'])
def create_data():
    """
    RESTful endpoint to train the machine learning model.
    """
    try:
        train_machine_learning_model()
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 500 Internal Server Error


# POST /api/predictions
@api_bp.route('/predictions', methods=['POST'])
def create_prediction():
    """
    RESTful endpoint to generate a prediction from a model.
    """
    try:
        prediction = predict()
        return jsonify({'prediction': prediction}), 200  # 200 OK
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 500 Internal Server Error
    

app.register_blueprint(api_bp)
