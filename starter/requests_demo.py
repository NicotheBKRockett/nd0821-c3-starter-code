import pandas as pd
import requests
import json

url = 'https://nd0821-c3-starter-code-0lqu.onrender.com/data_inference/'

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
r1 = requests.get('https://nd0821-c3-starter-code-0lqu.onrender.com')
print(r1.status_code)
print(r1.json())
assert r1.status_code == 200

data=json.dumps(data)
r = requests.post(url, data=data)

print(r.status_code)
print(r.json)
assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
