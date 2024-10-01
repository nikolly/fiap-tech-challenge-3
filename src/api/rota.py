from flask import Flask, request, Blueprint, jsonify
from src.core.train_model import train_machine_learning_model
from flask_cors import CORS
import pickle
import os
import pandas as pd



colunas = ['temp_max', 'temp_afternoon']
modelo_path = 'modelo/modelo_regressao_linear.pkl'
modelo = pickle.load(open(modelo_path,'rb')) if os.path.exists(modelo_path) else None

app = Flask(__name__)
CORS(app)
api_bp = Blueprint('api', __name__, url_prefix='/api')


# POST /api/train
@api_bp.route('/train', methods=['POST'])
def create_data():
    """
    RESTful endpoint to train the machine learning model.
    """
    global modelo
    try:
        response = train_machine_learning_model()
        if response[1] == 200: 
            modelo = pickle.load(open('modelo/modelo_regressao_linear.pkl','rb'))
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 500 Internal Server Error


# POST /api/predictions
@api_bp.route('/prediction', methods=['POST'])
def create_prediction():
    """
    RESTful endpoint to generate a prediction from a model.
    """
    
    if not modelo:
        return jsonify({'error': 'Model not trained'}), 500
    
    try:
        dados = request.get_json()
        if not dados:
            return jsonify({'error': 'No data provided'}), 400
        
        # Transforme os dados em um DataFrame
        dados_input = pd.DataFrame([dados], columns=colunas)
        
        # Faça a previsão
        humidity = modelo.predict(dados_input)
        return jsonify({'humidity': round(humidity[0], 1)})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500  # 500 Internal Server Error
    

app.register_blueprint(api_bp)
