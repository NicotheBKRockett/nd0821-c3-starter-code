import pandas as pd
import requests
import pandas as pd

url = 'https://nd0821-c3-starter-code-7d5l.onrender.com/data_inference/'

csvfile = pd.read_csv("C:\\Users\\nicol\OneDrive\Bureau\Project3_NicolasDelay\\nd0821-c3-starter-code\starter\starter\cleaned_data_dropna.csv")

data = csvfile.iloc[1]
data = data[:-1].to_dict()
print(data)

r = requests.post(url, data=data)

print(r)
#assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())

