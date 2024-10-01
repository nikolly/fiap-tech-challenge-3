# FIAP Tech Challenge 3

This project is part of FIAP’s Machine Learning Engineering challenge. It provides a **REST API** built with **Flask** that predicts **humidity** based on meteorological data. The machine learning model is trained using linear regression with `statsmodels`, and the project also interacts with **AWS S3** for data storage.

## Table of Contents

- [FIAP Tech Challenge 3](#fiap-tech-challenge-3)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
    - [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
    - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Configuration](#configuration)
    - [Step 1: Configure your AWS Credentials](#step-1-configure-your-aws-credentials)
  - [Usage](#usage)
    - [Step 1: Run the main script](#step-1-run-the-main-script)
    - [Step 2: Run frontend](#step-2-run-frontend)
    - [Step 3: Train the Model](#step-3-train-the-model)
    - [Step 4: Make Predictions](#step-4-make-predictions)
  - [Project Structure](#project-structure)
  - [Testing](#testing)

## Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.10.x**: Make sure it's properly installed.
- **Pip**: Python package manager (comes with Python).
- **Virtualenv**: Recommended for managing project dependencies.
- **AWS Accont**: - An active AWS account (to access S3 and manage your credentials)

## Installation

### Step 1: Clone the Repository

Start by cloning the project to your local machine:

```bash
git clone https://github.com/nikolly/fiap-tech-challenge-3.git
cd fiap-tech-challenge-3
```

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to create a virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
# OR
venv\Scripts\activate      # On Windows
```

### Step 3: Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

### Step 1: Configure your AWS Credentials

Make sure to configure your AWS credentials. You can do this by setting up the ~/.aws/credentials file with your access credentials:

```bash
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
aws_session_token= YOUR_SECRET_TOKEN
```

## Usage

### Step 1: Run the main script

The API will be available at <http://127.0.0.1:5000>.

```bash
python main.py
```

### Step 2: Run frontend

This project includes a basic frontend built with HTML, CSS, and JavaScript that allows users to interact with the prediction endpoint easily. The frontend provides a user-friendly interface to input temperature data and receive humidity predictions directly.

The frontend will be available at <http://127.0.0.1:8000>.

```bash
cd frontend
python -m http.server 8000
```

### Step 3: Train the Model

If you need to train the machine learning model, send a POST request to the /api/train endpoint. This will download data from S3, train the model, and save it to the modelo directory.
Use curl or any HTTP client like Postman:

```bash
curl -X POST http://127.0.0.1:5000/api/train
```

**Response**:

```bash
{
  "message": "Model trained successfully"
}
```

### Step 4: Make Predictions

To get a humidity prediction based on temperature data, make a POST request to /api/prediction. Include the following JSON body:

```bash
{
  "temp_max": 30.5,
  "temp_afternoon": 28.4
}
```

Example curl command:

```bash
curl -X POST http://127.0.0.1:5000/api/prediction \
    -H "Content-Type: application/json" \
    -d '{"temp_max": 30.5, "temp_afternoon": 28.4}'
```

**Response:**

```bash
{
  "humidity": 45.7
}
```

## Project Structure

The project is organized as follows:

```bash
.
├── main.py                     # Main Flask application
├── src
│   ├── api
│   │   ├── routes.py           # The endpoints for the machine learning model
│   ├── core
│   │   ├── train_model.py      # Logic for training the machine learning model
│   │   └── load_model.py       # Helper functions, such as downloading data from S3
│   ├── function
│   │   └── functions.py        # Helper functions, such as downloading data from S3
├── tests
│   ├── test_train_model.py     # Unit tests for model training
├── frontend
│   ├── index.html              # Simple frontend to interact with the prediction API
├── modelo                      # Directory where the trained model is saved
├── data                        # Local folder to store downloaded data from S3
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Testing

Unit tests are located in the tests directory. To run the tests:

```bash
pytest tests/
```

This will execute the unit tests for the model training process and verify that everything is functioning as expected.
