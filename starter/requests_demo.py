import pandas as pd
import requests
import json
import os


from starter.ml.data import process_data

url = 'https://nd0821-c3-starter-code-7d5l.onrender.com/data_inference/'

#cwd = os.getcwd()
#print(cwd)

#data = pd.read_csv(cwd+'\starter\cleaned_data_dropna.csv')
#data = data.drop(columns=["Unnamed: 0"])

#data = data[:-1].iloc[2].to_dict()
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

