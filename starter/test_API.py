from fastapi.testclient import TestClient

from main import app


client = TestClient(app)

def test_say_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello dear user! Welcome at the third project of the Udacity ML DevOps Nanodegree Program. I am Nicolas Delay"}


def test_inference_data_high():

    r = client.post("/data_inference/", json={
        "age": 43,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 6723,
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    print(r)
    assert r.status_code == 200
    #assert r.json() == {"response": "1"}

def test_inference_data_low():
    r = client.post("/data_inference/", json={
        "age": 39,
        "workclass": "Private",
        "fnlgt": 83311,
        "education": "HS-grad",
        "education_num" : 9,
        "marital_status": "Divorced",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain":0,
        "capital-loss":0,
        "hours_per_week": 30,
        "native_country": "United-States"
    })
    assert r.status_code == 200
    #assert r.json() == {"response": "0"}

