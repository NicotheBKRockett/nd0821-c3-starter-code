# Put the code for your API here.
#print("Test 7")

from typing import Union
from pydantic import BaseModel
import pickle

#def foo(a: Union[list,str], b: int = 5) -> str:
#    pass

from fastapi import FastAPI

try:
    from starter.ml.model import inference
    from starter.ml.data import process_data
except ImportError:
    # Handle the case where the imports might be nested differently
    from starter.starter.ml.model import inference
    from starter.starter.ml.data import process_data

class Item(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlwgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
                "age": 43,
                "workclass": 'State-gov',
                "fnlgt": 77516,
                "education": 'Bachelors',
                "education_num": 17,
                "marital_status": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "FMale",
                "capital_gain": 2000,
                "capital_loss": 0,
                "hours_per_week": 35,
                "native_country": 'United-States'
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
    X, y, encoder, lb = process_data(data, categorical_features=[], label=None, training=True, encoder=None, lb=None)
    predictions = inference(model, X)
    return {'response': predictions}

