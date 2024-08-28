from fastapi.testclient import TestClient

from main import app
import json
import pandas as pd
import os


client = TestClient(app)

def test_say_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello dear user! Welcome at the third project of the Udacity ML DevOps Nanodegree Program. I am Nicolas Delay"}


def test_inference_data_high():

    data = {
        'age': 33,
        'workclass': 'Local-gov',
        'fnlgt': 198183,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Prof-specialty',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': 'United-States'
    }
    request = client.post("/data_inference/", data=json.dumps(data))
    assert request.status_code == 200
    #assert request.json() == " >50K"


def test_inference_data_low():
    data = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    request = client.post("/data_inference/", data=json.dumps(data))
    assert request.status_code == 200
    #assert request.json() == " <=50K"





