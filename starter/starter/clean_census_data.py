import os
import pandas as pd

directory = os.getcwd()
path = directory.replace("starter\starter","starter\data\census.csv")
cleaned_data = pd.read_csv(path,skipinitialspace = True)

columns = [*cleaned_data.columns]

for column in columns:
    cleaned_data[column].replace("?",None,inplace = True)

#print(cleaned_data.info())

cleaned_data_dropna = cleaned_data.dropna()

cleaned_data_dropna.info()
cleaned_data_dropna.to_csv(os.path.join(directory,"cleaned_data_dropna.csv"))
