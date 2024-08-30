from fastapi.testclient import TestClient
from main import app
import json
import pickle
try:
    from starter.ml.model import compute_model_metrics, inference
except ImportError:
    from starter.starter.ml.model import compute_model_metrics, inference


client = TestClient(app)

def test_say_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello dear user! Welcome at the third project of the Udacity ML DevOps Nanodegree Program. I am Nicolas Delay"}

def test_inference_data_high():

    data = {
        'age': 52,
        'workclass': 'Private',
        'fnlgt': 78654,
        'education': 'Doctorate',
        'education_num': 13,
        'marital_status': 'Married',
        'occupation': 'Exec-managerial',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': 'United-States'
    }
    request = client.post("/data_inference/", data=json.dumps(data))
    assert request.status_code == 200
    assert request.json() == {'response': [1]}

def test_inference_data_low():
    data = {
        'age': 22,
        'workclass': 'State-gov',
        'fnlgt': 42160,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Divorced',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'Black',
        'sex': 'Female',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 25,
        'native_country': 'United-States'
    }
    request = client.post("/data_inference/", data=json.dumps(data))
    assert request.status_code == 200
    assert request.json() == {'response': [0]}


def test_accuracy():
    filename = "my_model.pkl"
    model = pickle.load(open(filename, "rb"))

    pickle_file_name = 'starter/saved_variables.pkl'
    data = pickle.load(open(pickle_file_name, "rb"))

    X = data['X_test']
    y = data['y_test']

    predictions_X_test = inference(model, X)
    precision, _, _ = compute_model_metrics(y, predictions_X_test)

    assert precision >= 0.7

def test_recall():
    filename = "my_model.pkl"
    model = pickle.load(open(filename, "rb"))

    pickle_file_name = 'starter/saved_variables.pkl'
    data = pickle.load(open(pickle_file_name, "rb"))

    X = data['X_test']
    y = data['y_test']

    predictions_X_test = inference(model, X)
    _, recall, _ = compute_model_metrics(y, predictions_X_test)

    assert recall >= 0.6

def test_fbeta():
    filename = "my_model.pkl"
    model = pickle.load(open(filename, "rb"))

    pickle_file_name = 'starter/saved_variables.pkl'
    data = pickle.load(open(pickle_file_name, "rb"))

    X = data['X_test']
    y = data['y_test']

    predictions_X_test = inference(model, X)
    _, _, fbeta = compute_model_metrics(y, predictions_X_test)

    assert fbeta >= 0.6







