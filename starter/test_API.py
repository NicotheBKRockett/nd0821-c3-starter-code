from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def say_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello dear user! Welcome at the third project of the Udacity ML DevOps Nanodegree Program. I am Nicolas Delay"}


def test_inference_data_high():

    r = client.post("/", json={
        "age": 60,
        "workclass": "Private",
        "education": "Doctorate",
        "maritalStatus": "Never-married",
        "occupation": "Transport-moving",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "hoursPerWeek": 80,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"response": "1"}

def test_inference_data_low():
    r = client.post("/", json={
        "age": 23,
        "workclass": "Private",
        "education": "HS-grad",
        "maritalStatus": "Divorced",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 30,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"response": "0"}

