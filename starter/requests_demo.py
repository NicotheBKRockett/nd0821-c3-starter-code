import pandas as pd
import requests
import json
import os

url = 'https://nd0821-c3-starter-code-7d5l.onrender.com/data_inference/'

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

data=json.dumps(data)
r = requests.post(url, data=data)

print(r.status_code)
assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
