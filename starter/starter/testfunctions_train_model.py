import pytest
import logging
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score

logging.basicConfig(filename='pytests.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
def data():
    pickle_file_name = 'saved_variables.pkl'
    with open(pickle_file_name, 'rb') as file:
        loaded_variables = pickle.load(file)

    X_train = loaded_variables['X_train']
    X_test = loaded_variables['X_test']
    y_train = loaded_variables['y_train']
    y_test = loaded_variables['y_test']

    return  X_train, X_test, y_train, y_test

def test_loaddata():
    filename = "my_model.pickle"
    try:
        model = pickle.load(open(filename, "rb"))
    except:
        logger.info('FAILED test_loaddata: Error occured during loading model')
    return
def test_metrics(data):
    filename = "my_model.pickle"
    model = pickle.load(open(filename, "rb"))
    logger.info('SUCCES test_metrics: Data and model loaded')
    X_train, X_test, y_train, y_test = data
    predictions_X_test = model.predict(X_test)
    fbeta = fbeta_score(y_test, predictions_X_test, beta=1, zero_division=1)
    precision = precision_score(y_test, predictions_X_test, zero_division=1)
    recall = recall_score(y_test, predictions_X_test, zero_division=1)
    assert precision >= 0.93
    logger.info('SUCCES test_metrics: Precision >= 0.93')
    assert recall <= 0.2
    logger.info('SUCCES test_metrics: Recall <= 0.2')
    assert fbeta <= 0.2
    logger.info('SUCCES test_metrics: fbeta <= 0.2')
    return

def test_traintestratio(data):
    X_train, X_test, y_train, y_test = data

    assert len(X_test)/(len(X_test)+len(X_train)) < 0.22
    assert len(X_test) / (len(X_test) + len(X_train)) > 0.18
    logger.info('SUCCES test_traintestratio')

