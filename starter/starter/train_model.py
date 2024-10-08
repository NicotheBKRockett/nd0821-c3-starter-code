# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ydata_profiling import ProfileReport
import pickle


from sklearn.preprocessing import OneHotEncoder
# Add code to load in the data.
data = pd.read_csv("cleaned_data_dropna.csv")

#print(data.info())
# Optional enhancement, use K-fold cross validation instead of a train-test split.

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

save = False

if save:
    variables_to_save = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    # Specify the file name
    pickle_file_name = 'saved_variables.pkl'

    # Save the variables to a pickle file
    with open(pickle_file_name, 'wb') as file:
        pickle.dump(variables_to_save, file)

# Train and save a model.
train = False
filename = "my_model.pickle"

if train == True:
    model = train_model(X_train,y_train)

    # save model
    pickle.dump(model, open(filename, "wb"))
else:
    model = pickle.load(open(filename, "rb"))

predictions_X_test = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions_X_test)
print(precision, recall, fbeta)
