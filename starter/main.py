# Put the code for your API here.
#print("Test 7")

from typing import Union
from pydantic import BaseModel

#def foo(a: Union[list,str], b: int = 5) -> str:
#    pass

from fastapi import FastAPI

class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello dear user! Welcome at the third project of the Udacity ML DevOps Nanodegree Program. I am Nicolas Delay"}


@app.post("/data/")
async def ingest_data(data: Data):
    if data.feature_1 < 0:
        raise HTTPException(status_code=400, detail="feature_1 needs to be above 0.")
    if len(data.feature_2) > 280:
        raise HTTPException(
            status_code=400,
            detail=f"feature_2 needs to be less than 281 characters. It has {len(data.feature_2)}.",
        )
    return data