import pandas as pd
from src.function.functions import download_data_from_s3, parquet_to_json


def train_machine_learning_model():
    try:
        bucket_name = 'openweather-tc3'
        s3_folder = 'Silver'
        local_folder = 'data'
        
        download_data_from_s3(bucket_name, s3_folder, local_folder)
        
        ...        
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
