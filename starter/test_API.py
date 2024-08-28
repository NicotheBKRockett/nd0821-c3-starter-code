from fastapi.testclient import TestClient

from main import app
import pandas as pd
import os


client = TestClient(app)

def test_say_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello dear user! Welcome at the third project of the Udacity ML DevOps Nanodegree Program. I am Nicolas Delay"}


def test_inference_data_high():
    #cwd = os.getcwd()
    #data = pd.read_csv(cwd + '\starter\cleaned_data_dropna.csv')

    #data = data.iloc[2][1:-1].to_dict()

    #r = client.post("/data_inference/", data= data)
    #print(r)

    r = client.post("/data_inference/", json={
        "age": 60,
        "workclass": "Private",
        'fnlgt': 77516,
        "education": "Doctorate",
        "education_num": 6723,
        "marital_status": "Divorced",
        "occupation": "Transport-moving",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hours_per_week": 76,
        "capital_gain": 0,
        "capital_loss":0,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"response": "1"}

def test_inference_data_low():
    #cwd = os.getcwd()
    #data = pd.read_csv(cwd + '\starter\cleaned_data_dropna.csv')

    #data = data.iloc[2][1:-1].to_dict()

    #r = client.post("/data_inference/", data = data)
    #print(r)

    r = client.post("/data_inference/", json={
        "age": 20,
        "workclass": "Private",
        'fnlgt': 77516,
        "education": "11th",
        "education_num": 6723,
        "marital_status": "Never_married",
        "occupation": "Transport-moving",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "hours_per_week": 20,
        "capital_gain": 0,
        "capital_loss": 0,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    #assert r.json() == {"response": "0"}


