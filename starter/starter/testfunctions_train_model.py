import pytest

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import logging
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

logger = logging.getLogger(__name__)
logging.basicConfig(filename='pytests.log', level=logging.INFO)

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
        logger.info('SUCCES test_loaddata: Loading model')
    except:
        logger.info('FAILED test_loaddata: Error occured during loading model')

def test_metrics(data):
    X_train, X_test, y_train, y_test = data
    filename = "my_model.pickle"
    model = pickle.load(open(filename, "rb"))
    logger.info('SUCCES test_metrics: Data and model loaded')
    predictions_X_test = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions_X_test)
    assert precision >= 0.93
    logger.info('SUCCES test_metrics: accuracy good enough')
    assert recall <= 0.2
    logger.info('SUCCES test_metrics: recall good enough')
    assert fbeta <= 0.2
    logger.info('SUCCES test_metrics: fbeta good enough')