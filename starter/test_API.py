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
    cwd = os.getcwd()
    data = pd.read_csv(cwd + '\starter\cleaned_data_dropna.csv')

    data = data.iloc[2][1:-1].to_dict()

    r = client.post("/data_inference/", data= data)
    print(r)
    assert r.status_code == 200
    #assert r.json() == {"response": "1"}

def test_inference_data_low():
    cwd = os.getcwd()
    data = pd.read_csv(cwd + '\starter\cleaned_data_dropna.csv')

    data = data.iloc[2][1:-1].to_dict()

    r = client.post("/data_inference/", data = data)
    print(r)
    assert r.status_code == 200
    #assert r.json() == {"response": "0"}


