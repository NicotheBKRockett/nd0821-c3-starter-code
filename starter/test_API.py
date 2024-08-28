from fastapi.testclient import TestClient
from main import app
import json
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
    print(request.status_code)
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
    print(request.status_code)
    assert request.status_code == 200
    assert request.json() == {'response': [0]}





