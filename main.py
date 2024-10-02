from flask import Flask
from flask_cors import CORS
from src.api.routes import api_bp
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Register blueprint
app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(debug=True)
