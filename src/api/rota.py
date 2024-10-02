from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from src.core.train_model import train_machine_learning_model
from flask_cors import CORS
import pickle
import os
import pandas as pd

# Configurações iniciais
colunas = ['temp_max', 'temp_afternoon']
modelo_path = 'modelo/modelo_regressao_linear.pkl'
modelo = pickle.load(open(modelo_path, 'rb')) if os.path.exists(modelo_path) else None

# Inicializando a aplicação Flask e a extensão Flask-RESTx
app = Flask(__name__)
CORS(app)

# Alterar o caminho da documentação para '/docs' (padrão)
api = Api(app, version='1.0', title='Machine Learning API',
          description='API para treinar modelos e gerar previsões de umidade',
          doc='/docs')  # Endpoint para documentação Swagger

# Definindo um namespace para as rotas da API
ns = api.namespace('api', description='Operações relacionadas ao modelo de Machine Learning')

# Modelo de entrada para previsões
input_model = api.model('InputModel', {
    'temp_max': fields.Float(required=True, description='Temperatura máxima'),
    'temp_afternoon': fields.Float(required=True, description='Temperatura da tarde')
})

# Modelo de saída para a previsão de umidade
output_model = api.model('OutputModel', {
    'humidity': fields.Float(description='Previsão de umidade')
})

# Modelo de resposta para treinamento
train_response_model = api.model('TrainResponse', {
    'message': fields.String(description='Mensagem de sucesso ou erro')
})

# POST /api/train
@ns.route('/train')
class TrainModel(Resource):
    @ns.doc('train_model')
    @ns.marshal_with(train_response_model)
    def post(self):
        """
        Endpoint RESTful para treinar o modelo de aprendizado de máquina.
        """
        global modelo
        try:
            response = train_machine_learning_model()
            if response[1] == 200:
                modelo = pickle.load(open('modelo/modelo_regressao_linear.pkl', 'rb'))
            return response
        except Exception as e:
            api.abort(500, str(e))

# POST /api/predictions
@ns.route('/prediction')
class Prediction(Resource):
    @ns.doc('create_prediction')
    @ns.expect(input_model)
    @ns.marshal_with(output_model)
    def post(self):
        """
        Endpoint RESTful para gerar uma previsão a partir do modelo treinado.
        """
        global modelo
        if not modelo:
            api.abort(500, 'Model not trained')

        try:
            dados = request.get_json()
            if not dados:
                api.abort(400, 'No data provided')

            # Transformando os dados em um DataFrame
            dados_input = pd.DataFrame([dados], columns=colunas)

            # Fazendo a previsão
            humidity = modelo.predict(dados_input)
            return {'humidity': round(humidity[0], 1)}
        except Exception as e:
            api.abort(500, str(e))


if __name__ == '__main__':
    app.run(debug=True)
