import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

@pytest.fixture
def data():
    data = pd.read_csv("cleaned_data_dropna.csv")

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return  X_train, X_test, y_train, y_test

def test_loaddata():
    filename = "my_model.pickle"
    try:
        model = pickle.load(open(filename, "rb"))
    except:
        print('Error occured during loading model')
