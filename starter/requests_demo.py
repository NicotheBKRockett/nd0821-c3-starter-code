import pandas as pd
import requests
import pandas as pd
import os

from starter.ml.data import process_data

url = 'https://nd0821-c3-starter-code-7d5l.onrender.com/data_inference/'

cwd = os.getcwd()
print(cwd)

data = pd.read_csv(cwd+'\starter\cleaned_data_dropna.csv')

data = data.iloc[2][1:-1].to_dict()
print(data)

r = requests.post(url, data=data)

print(r)
assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())

