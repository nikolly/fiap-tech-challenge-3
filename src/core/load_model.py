import pickle
import os
import logging

def load_model(model_path: str):
    """
    Load the trained machine learning model from the specified path.

    Args:
        model_path (str): Path to the pickle file containing the trained model.

    Returns:
        The loaded model object or None if loading fails.
    """
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
                logging.info("Model loaded successfully")
                return model
        else:
            logging.warning(f"Model file does not exist at path: {model_path}")
            return None
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None
