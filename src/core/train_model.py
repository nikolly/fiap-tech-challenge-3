import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from src.function.functions import download_data_from_s3
import pickle
from typing import List, Tuple, Dict, Any
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError


def validate_data(record: Dict[str, Any]) -> bool:
    required_fields = ['temp_max', 'temp_afternoon', 'humidity_afternoon']
    for field in required_fields:
        if field not in record:
            logging.warning(f"Missing field '{field}' in record: {record}")
            return False
        if not isinstance(record[field], (int, float)):
            logging.warning(f"Invalid type for field '{field}' in record: {record}")
            return False
    return True


def get_data_from_files(local_folder: str) -> List[Dict[str, Any]]:
    data = []
    for file_name in os.listdir(local_folder):
        file_path = os.path.join(local_folder, file_name)
        if file_name.endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    if content and validate_data(content):
                        data.append(content)
            except json.JSONDecodeError:
                logging.warning(f"Failed to decode JSON in file: {file_path}")
            except Exception as e:
                logging.error(f"Unexpected error reading file {file_path}: {e}")
        else:
            logging.info(f"Skipping non-JSON file: {file_path}")
    return data


def train_machine_learning_model(bucket_name: str, s3_folder: str, local_folder: str, model_path: str) -> Tuple[Dict[str, Any], int]:
    """
    Train a machine learning model using data from S3 and save the model as a pickle file.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_folder (str): Folder path in the S3 bucket.
        local_folder (str): Local directory to store downloaded data.
        model_path (str): Path to save the trained model.

    Returns:
        Tuple containing a JSON response and an HTTP status code.
    """
    try:        
        # Download data from S3
        download_data_from_s3(bucket_name, s3_folder, local_folder)
        data = get_data_from_files(local_folder)
        if not data:
            raise ValueError("No valid data found for training.")
        
        df = pd.DataFrame(data)
        
        # Validate required columns
        required_columns = ['temp_max', 'temp_afternoon', 'humidity_afternoon']
        if not all(column in df.columns for column in required_columns):
            missing = list(set(required_columns) - set(df.columns))
            raise ValueError(f"Missing columns in data: {missing}")

        # Split data into features and target
        y = df['humidity_afternoon']
        x = df.drop(columns='humidity_afternoon')

        # Split dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Create a DataFrame for training
        df_train = pd.DataFrame(data= x_train)
        df_train['humidity_afternoon'] = y_train

        # Train linear regression model using OLS (Ordinary Least Squares)
        model = ols('humidity_afternoon ~ temp_max + temp_afternoon', data= df_train).fit()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the trained model as a pickle file
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)

        logging.info("Model trained and saved successfully")
        return {'message': 'Model trained successfully'}, 200
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return {'error': 'Required file not found.'}, 404
    except (NoCredentialsError, PartialCredentialsError):
        logging.error("AWS credentials issue")
        return {'error': 'AWS credentials not available or incomplete.'}, 403
    except ValueError as e:
        logging.error(f"Value error: {e}")
        return {'error': str(e)}, 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {'error': 'An unexpected error occurred.'}, 500
