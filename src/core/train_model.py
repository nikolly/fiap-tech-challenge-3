import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from src.function.functions import download_data_from_s3
import pickle
from flask import jsonify


def train_machine_learning_model():
    try:
        bucket_name = 'openweather-tc3'
        s3_folder = 'Silver'
        local_folder = 'data'
        
        # download_data_from_s3(bucket_name, s3_folder, local_folder)
        dados = get_data_from_files(local_folder)
        df = pd.DataFrame(dados)

        y = df['humidity_afternoon']
        x = df.drop(columns='humidity_afternoon')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        df_train = pd.DataFrame(data= x_train)
        df_train['humidity_afternoon'] = y_train

        modelo = ols('humidity_afternoon ~ temp_max + temp_afternoon', data= df_train).fit()
                
        caminho_arquivo = 'modelo/modelo_regressao_linear.pkl'
        diretorio = os.path.dirname(caminho_arquivo)

        # Verificar se o diretório existe, se não, criar
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)

        # Salvar o modelo em um arquivo usando pickle
        with open(caminho_arquivo, 'wb') as arquivo:
            pickle.dump(modelo, arquivo)
        
        data = {'message': 'Model trained successfully'}
        return jsonify(data), 200
        
    except Exception as e:
        print(e)
        data = {'error': str(e)}
        return jsonify(data), 500
    

def get_data_from_files(local_folder):
    dados = []
    for arquivo in os.listdir(local_folder):
        caminho_arquivo = os.path.join(local_folder, arquivo)
        with open(caminho_arquivo, 'r') as f:
            try:
                conteudo = json.load(f)
                if conteudo:
                    dados.append(conteudo)
            except json.JSONDecodeError:
                print(f"Erro ao decifrar o JSON no arquivo {caminho_arquivo}")

    return dados
