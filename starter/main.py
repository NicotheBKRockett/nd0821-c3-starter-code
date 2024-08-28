import pandas as pd
from pydantic import BaseModel
import pickle
from joblib import load
from fastapi import FastAPI

try:
    from starter.ml.model import inference
    from starter.ml.data import process_data
except ImportError:
    # Handle the case where the imports might be nested differently
    from starter.starter.ml.model import inference
    from starter.starter.ml.data import process_data

class Item(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
            }
        }


# Instantiate the app.
app = FastAPI()
# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello dear user! Welcome at the third project of the Udacity ML DevOps Nanodegree Program. I am Nicolas Delay"}

@app.post("/data_inference/")
async def inference_data(data: Item):
    filename = "my_model.pickle"
    model = pickle.load(open(filename, "rb"))
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    encoder = load("starter/encoder.joblib")
    lb = load("starter/lb.joblib")

    data = pd.DataFrame([data.dict()])

    X, _, _, _ = process_data(data, cat_features, label=None, training=False, encoder=encoder, lb=lb)
    prediction = inference(model, X)
    prediction_list = prediction.tolist()
    return {'response': prediction_list}

