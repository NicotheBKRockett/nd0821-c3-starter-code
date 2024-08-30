import joblib
from sklearn.model_selection import train_test_split
import pickle
# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference

data = pd.read_csv("cleaned_data_dropna.csv")
data = data.drop(columns=["Unnamed: 0"])
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


data_train, data_test  = train_test_split(data, test_size=0.2, random_state=42)

education_possibilities = set(data['education'])
education_possibilities = [*education_possibilities]
#print(education_possibilities)
encoder = joblib.load('encoder.joblib')
lb = joblib.load('lb.joblib')
filename = "my_model.pkl"
model = pickle.load(open(filename, "rb"))
dict = {}

for education in education_possibilities:
    #print(education)
    data_temp = data_test[data_test['education'] == education]
    #print(data_temp.shape)
    X, y, _, _ = process_data(data_temp, categorical_features=cat_features, label = 'salary', lb = lb,training=False, encoder = encoder)
    prediction = inference(model,X)
    precision, recall, fbeta = compute_model_metrics(y, prediction)
    dict[education] = {'precision': precision, 'recall': recall, 'fbeta': fbeta}

with open('metrics_slices_data.pkl', 'wb') as f:
    pickle.dump(dict, f)
